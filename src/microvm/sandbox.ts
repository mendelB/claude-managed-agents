import { Sandbox as SandboxBase, getSandbox } from "@cloudflare/sandbox";
import type { DirectoryBackup } from "@cloudflare/sandbox";
import Anthropic from "@anthropic-ai/sdk";
import { toErrorMessage } from "../helpers";
import { applyEgressPolicy } from "../egress/handler";
import { resolveSessionPolicy } from "../egress/resolve";
import type { CompiledPolicy } from "../egress/types";
import { buildCfTools } from "../tools/cf";
import { runCustomToolDispatcher } from "../isolate/custom-dispatch";
import { isolateBrowserTools } from "../isolate/tools";
import { buildMicrovmStockTools } from "./stock-tools";
import { runHeartbeatLoop } from "../heartbeat";
import { ANTHROPIC_BETA } from "../anthropic";

// How long the MicroVM container stays alive after the agent goes idle
// before the base Container class snapshots /workspace and stops it.
// Short by default so we don't keep MicroVMs warm longer than needed;
// override per deployment in wrangler.jsonc / sandbox.ts if your sessions
// burst back to life often and you'd rather pay for warmth than cold
// boots.
export const SESSION_IDLE_TTL = "3m";

// Workspace directory the snapshots cover. The Sandbox SDK enforces that
// backup paths live under one of /workspace, /home, /tmp, /var/tmp, or
// /app. We snapshot only /workspace because that's where user code +
// agent-authored files live; /home and /tmp churn too much to be worth
// archiving and would inflate snapshot size for no benefit.
const SNAPSHOT_DIR = "/workspace";

// Storage key for the most-recent DirectoryBackup handle. Persisted to DO
// storage so it survives hibernation; restored on the next cold-start
// dispatch. The handle is a tiny serialisable object — id + dir + a flag
// — so it's cheap to keep around indefinitely.
const SNAPSHOT_KEY = "latest_snapshot";

// Backup TTL — 7 days. R2 lifecycle rules on the BACKUP_BUCKET should be
// set to garbage-collect under the `backups/` prefix beyond this. Each
// `createBackup` call writes a fresh archive, so the TTL just bounds how
// long a stale snapshot can sit there before being GC'd.
const SNAPSHOT_TTL_SECONDS = 7 * 24 * 60 * 60;

export interface DispatchOpts {
  sessionId: string;
  workId: string;
  environmentId: string;
  baseURL: string;
}

// One Sandbox per session. The container is started by the Sandbox SDK
// (its own entrypoint serves the platform's HTTP API for exec /
// readFile / writeFile / etc.) and is then driven entirely from this
// DO via that API — there is no in-container worker process.
//
// How we differ from the Anthropic self-hosted-sandboxes guide
// ------------------------------------------------------------
// The recommended worker-side architecture
// (https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes#environment-worker)
// for a webhook-triggered setup is to use the SDK's
// `EnvironmentWorker.handleItem()` per claimed work item, or for the
// always-on / container-per-session variant, to use `ant beta:worker
// run` as the container's `ENTRYPOINT`. Both helpers own claim →
// skills download → tool dispatch → result post → stop in one
// process.
//
// Neither helper runs in a Cloudflare Worker Durable Object: the SDK
// helper requires Node + `/bin/bash`; `ant beta:worker run` is a
// separate process. So we own the equivalent protocol ourselves and
// compose three pieces inside the DO:
//
//   1. `runCustomToolDispatcher` (src/isolate/custom-dispatch.ts) —
//      reads the session event stream and answers BOTH
//      `agent.tool_use` (via `stockTools` built by
//      `buildMicrovmStockTools` — they `exec`/`readFile`/`writeFile`
//      inside the container) AND `agent.custom_tool_use` (via the
//      worker-binding-backed `tools` — cf_* family + user customs).
//   2. `runHeartbeatLoop` (src/heartbeat.ts) — holds the work-item
//      lease the docs say `EnvironmentWorker` would hold for us, and
//      signals the dispatcher to wind down on `state: stopping`.
//   3. `work.stop({ force: true })` on exit — the equivalent of
//      `handleItem()` exiting cleanly.
//
// We previously ran `ant beta:worker run` as a child process inside
// the container alongside this dispatcher. The two were supposedly on
// disjoint event types (the in-container CLI handled `agent.tool_use`,
// the DO handled `agent.custom_tool_use`), but on self-hosted-only
// envs the work-queue path the CLI polls stopped surfacing tool
// turns, so every built-in `write` hung indefinitely. Removing it
// gives us one clear owner per session and avoids the dual-claim race
// where both processes try to heartbeat the same work item.
//
// Skills download is still a TODO — same limitation as the Isolate
// runner (see src/isolate/runner.ts around DISPATCHER_FIBER). The
// SDK's `setupSkills` uses Node `fs`/`path` directly; a
// Workers-compatible variant (`sandbox.writeFile` into
// `/workspace/skills/<name>/`) would slot in here.
//
// Egress proxy:
//   - `outbound` is the static catch-all fall-through. It MUST be defined
//     even if a policy isn't attached yet — `interceptHttps = true` makes
//     the SDK pass `SANDBOX_INTERCEPT_HTTPS=1` to the container, but the
//     ephemeral CA cert is only injected at `/etc/cloudflare/certs/...`
//     when the SDK detects a catch-all handler (`ctor.outbound !==
//     undefined` or a runtime `setOutboundHandler` override). Without
//     `outbound`, a PTY upgrade for a never-dispatched session boots a
//     container that refuses to start with "Certificate not found,
//     refusing to start without HTTPS interception enabled", and the
//     WebSocket dies with "Network connection lost".
//   - `outboundHandlers.policy` is the named handler we register on top
//     of the catch-all; it receives a CompiledPolicy via params and
//     calls applyEgressPolicy().
//   - On dispatch, we look up the matching EgressPolicy in KV, compile it
//     (resolving secrets), and call setOutboundHandler("policy", { policy }).
//     `outboundHandlerOverride` takes precedence over the static
//     `outbound` so the policy handler wins whenever one is attached.
export class Sandbox extends SandboxBase<Env> {
  override sleepAfter = SESSION_IDLE_TTL;
  // Intercept HTTPS so the egress proxy can see TLS traffic too — without
  // this, only port 80 traffic flows through the policy handler and HTTPS
  // bypasses egress entirely. The platform mounts a CA at
  // /etc/cloudflare/certs/cloudflare-containers-ca.crt and the sandbox
  // runtime auto-trusts it for curl/Node/Python/Git on startup.
  // https://developers.cloudflare.com/sandbox/guides/outbound-traffic/#https-traffic
  override interceptHttps = true;

  // Custom-tool dispatcher controller. Set when a dispatcher is running
  // against the session, cleared when it exits or is aborted. We use the
  // same AbortController pattern IsolateRunner does so a stop / drift
  // restart can wind the dispatcher down predictably.
  private customDispatchCtrl: AbortController | undefined;

  // In-flight cold-boot promise. `ensureStarted` is called from any path
  // that "turns on" the sandbox — dispatch on a webhook, PTY upgrade,
  // /exec API call. Concurrent callers MUST all block on the same
  // restore: if two arrive while the container is cold, one starts the
  // container + kicks off restore, and the other has to await the same
  // promise. Without this, a fast PTY upgrade racing with dispatch
  // could read or write /workspace before restoreBackup() finishes and
  // clobber the previous session's files.
  private bootPromise: Promise<void> | undefined;

  static {
    // Catch-all fall-through. Required so the SDK promotes the container
    // to intercept-all mode (`shouldInterceptAllOutbound() === true`),
    // which in turn calls `interceptOutboundHttps('*', fetcher)` BEFORE
    // `container.start()` and provisions the ephemeral CA at
    // `/etc/cloudflare/certs/cloudflare-containers-ca.crt`. Without this
    // handler, `interceptHttps = true` only sets the
    // `SANDBOX_INTERCEPT_HTTPS=1` env var on the container — the cert
    // never lands, and the sandbox runtime refuses to start with
    // "Certificate not found, refusing to start without HTTPS
    // interception enabled". A `setOutboundHandler('policy', ...)` call
    // from `applyPolicy()` overrides this at runtime for sessions that
    // have an attached EgressPolicy.
    Sandbox.outbound = async (req) => fetch(req);
    Sandbox.outboundHandlers = {
      // Params shape: { policy } — the policy handler routes the request
      // through applyEgressPolicy (allow/deny, header injection, VPC
      // routing, proxy fn). Auditing isn't built in; operators can layer
      // their own logging on top if they need a structured trail.
      policy: async (req, env, ctx) => {
        const params = ctx.params as { policy: CompiledPolicy } | undefined;
        // applyEgressPolicy is a pure function — it takes any record-
        // shaped env so the test harness can hand in a fake. The cast
        // here just widens our typed Env to that contract.
        return applyEgressPolicy(
          req,
          env as unknown as Record<string, unknown>,
          params,
        );
      },
    };
  }

  async isLive(): Promise<boolean> {
    return this.ctx.container?.running ?? false;
  }

  // Renew the container's activity timer mid-tool-call. Called from
  // the custom-tool dispatcher's `onInFlightChange` hook whenever the
  // in-flight count crosses into > 0 so a long-running `bash` doesn't
  // get its container reaped under `SESSION_IDLE_TTL` while it's still
  // doing useful work. Best-effort — failure is swallowed by the
  // caller because the standard activity path will eventually catch up.
  //
  // `setEnvVars({})` is the cheapest no-op RPC the Sandbox SDK exposes
  // that still touches the container's activity tracker. We can't
  // simply skip this and rely on the in-flight tool's own RPC because
  // tools that hand work off to long-running subprocesses (npm install
  // with output we're not streaming) can stop producing RPC traffic
  // for multi-minute stretches.
  async renewContainerActivity(): Promise<void> {
    if (!(await this.isLive())) return;
    try {
      await this.setEnvVars({});
    } catch (error) {
      console.warn(
        `[sandbox] activity renewal failed: ${toErrorMessage(error)}`,
      );
    }
  }

  async applyPolicy(policy: CompiledPolicy | null): Promise<void> {
    if (policy) {
      await this.setOutboundHandler("policy", { policy });
    }
    // When policy is null, we leave the default outbound (no-op) in place so
    // the sandbox can reach the internet without restrictions.
  }

  // True when a BACKUP_BUCKET R2 binding is present. We need it for
  // `localBucket: true` (dev) and we still set the binding in production
  // even though presigned-URL mode is the actual path used; the SDK
  // checks `env.BACKUP_BUCKET` exists at the start of every backup call.
  private hasBackupBinding(): boolean {
    return (
      typeof this.env.BACKUP_BUCKET === "object" &&
      this.env.BACKUP_BUCKET !== null
    );
  }

  // We use presigned-URL mode in production (faster, fewer binding hops)
  // and `localBucket: true` only when the SDK insists on it. The
  // detection mirrors `detectCredentials()` in the Sandbox SDK: any of
  // R2_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID being present means we have
  // creds to mint presigned URLs.
  private hasR2Credentials(): boolean {
    const {
      R2_ACCESS_KEY_ID,
      R2_SECRET_ACCESS_KEY,
      AWS_ACCESS_KEY_ID,
      AWS_SECRET_ACCESS_KEY,
    } = this.env;
    return Boolean(
      (R2_ACCESS_KEY_ID && R2_SECRET_ACCESS_KEY) ||
      (AWS_ACCESS_KEY_ID && AWS_SECRET_ACCESS_KEY),
    );
  }

  // Take a snapshot of /workspace and persist the handle to DO storage.
  // Called from onActivityExpired (auto-suspend) and from the explicit
  // /api/.../stop handler before destroy(). Best-effort: any error is
  // logged and swallowed so we never fail a stop because of a snapshot
  // failure.
  async snapshot(): Promise<DirectoryBackup | null> {
    if (!this.hasBackupBinding()) return null;
    if (!(await this.isLive())) {
      // Container is already down — nothing to snapshot. The previous
      // snapshot (if any) is still valid; leave the storage key alone.
      console.log(`[sandbox] snapshot skipped — container not running`);
      return null;
    }
    const useLocal = !this.hasR2Credentials();
    try {
      console.log(
        `[sandbox] snapshot start dir=${SNAPSHOT_DIR} mode=${useLocal ? "localBucket" : "presigned"}`,
      );
      const backup = await this.createBackup({
        dir: SNAPSHOT_DIR,
        ttl: SNAPSHOT_TTL_SECONDS,
        // Skip caches and node_modules — restoring them is faster than
        // re-downloading, but the archive size hit is large enough that
        // most users would rather skip them. Tunable via wrangler.jsonc
        // var if anyone needs to keep them.
        excludes: ["node_modules", ".cache", "*.log"],
        gitignore: true,
        ...(useLocal ? { localBucket: true } : {}),
      });
      await this.ctx.storage.put(SNAPSHOT_KEY, backup);
      console.log(`[sandbox] snapshot saved id=${backup.id}`);
      return backup;
    } catch (error) {
      // Logged as `error` (not `warn`) so the failure is easy to find
      // with a log query like `[sandbox] snapshot failed` — the
      // operator needs to know we couldn't capture /workspace and the
      // next cold boot will start empty.
      console.error(
        `[sandbox] snapshot failed dir=${SNAPSHOT_DIR} mode=${useLocal ? "localBucket" : "presigned"}: ${toErrorMessage(error)}`,
      );
      return null;
    }
  }

  // Restore the most-recent snapshot if one exists. Called from the
  // cold-boot path inside `ensureStarted` (which is in turn invoked by
  // dispatch, the PTY upgrade, and /exec). This method itself does
  // NOT check isLive — the caller is responsible for ensuring the
  // container is freshly booted before invoking. Wired this way to
  // make the "never overwrite a running container" invariant explicit
  // at the call site rather than buried inside this helper.
  //
  // Hard-timeboxed at 30s — if the SDK call stalls on a misconfigured
  // BACKUP_BUCKET (binding present but R2 creds wrong / missing) we'd
  // otherwise block dispatch indefinitely and the control plane would never
  // start, which manifests as every subsequent tool call (write, read,
  // bash, …) hanging from the agent's perspective.
  async restoreLatestSnapshot(): Promise<boolean> {
    if (!this.hasBackupBinding()) return false;
    const handle = await this.ctx.storage.get<DirectoryBackup>(SNAPSHOT_KEY);
    if (!handle) {
      console.log(`[sandbox] no snapshot to restore`);
      return false;
    }
    const timeoutMs = 30_000;
    const timeout = new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new Error(`restoreBackup timed out after ${timeoutMs}ms`)),
        timeoutMs,
      ),
    );
    try {
      console.log(
        `[sandbox] restoring snapshot id=${handle.id} dir=${handle.dir}`,
      );
      const result = await Promise.race([this.restoreBackup(handle), timeout]);
      console.log(
        `[sandbox] snapshot restore success=${result.success} id=${result.id} dir=${result.dir}`,
      );
      return result.success;
    } catch (error) {
      // Restore errors must not block dispatch indefinitely — log loudly
      // and fall through to a fresh /workspace. The handle stays in
      // storage so the operator can investigate; subsequent successful
      // snapshots will overwrite it. Logged as `error` (not `warn`) so
      // it stands out in production log queries like
      // `[sandbox] snapshot restore failed`.
      console.error(
        `[sandbox] snapshot restore failed handle=${handle.id} dir=${handle.dir} (continuing with empty workspace): ${toErrorMessage(error)}`,
      );
      return false;
    }
  }

  // Boot the container (idempotent) AND block on snapshot restore before
  // returning. Single entry point used by every code path that "turns
  // on" the sandbox for any reason: dispatch on a webhook, the PTY
  // upgrade at /ws/terminal, the /api/.../exec endpoint. Callers can
  // rely on /workspace being hydrated from the most-recent snapshot
  // (or fresh, if no snapshot exists / restore failed) once the
  // returned promise resolves.
  //
  // Concurrent callers share the same in-flight cold-boot promise via
  // `bootPromise` so two simultaneous "turn it on" requests can't race
  // to read /workspace mid-restore.
  async ensureStarted(envVars?: Record<string, string>): Promise<void> {
    if (await this.isLive()) return;
    if (this.bootPromise) {
      // Another caller is already booting the container + restoring.
      // Block on their promise so we observe the same "restore done"
      // ordering rather than touching /workspace mid-restore.
      await this.bootPromise;
      return;
    }
    const promise = (async () => {
      try {
        await this.start(envVars ? { envVars } : undefined);
      } catch (error) {
        console.error(
          `[sandbox] start() failed during ensureStarted: ${toErrorMessage(error)}`,
        );
        throw error;
      }
      // Cold boot path. We block work here — no caller of ensureStarted
      // returns until the restore attempt finishes, so the agent /
      // terminal / exec can't read or write /workspace before the
      // previous session's files are back.
      try {
        await this.restoreLatestSnapshot();
      } catch (error) {
        // restoreLatestSnapshot catches its own errors and returns
        // false, so reaching this branch means a bug rather than a
        // routine restore failure. Surface it loudly but don't refuse
        // to boot — a session with an empty workspace is more useful
        // than one that can't start at all.
        console.error(
          `[sandbox] unexpected error from restoreLatestSnapshot: ${toErrorMessage(error)}`,
        );
      }
    })();
    this.bootPromise = promise;
    try {
      await promise;
    } finally {
      if (this.bootPromise === promise) {
        this.bootPromise = undefined;
      }
    }
  }

  // Activity-expired alarm fires before the container is auto-suspended.
  // We snapshot /workspace first so the next cold boot can restore it,
  // then abort the custom-tool dispatcher so its polling loop exits and
  // the DO stops accruing active-duration cost. The base class then
  // stops the container (parent impl).
  override async onActivityExpired(): Promise<void> {
    try {
      await this.snapshot();
    } catch (error) {
      console.warn(
        `[sandbox] activity-expired snapshot threw: ${toErrorMessage(error)}`,
      );
    }
    this.customDispatchCtrl?.abort();
    return super.onActivityExpired();
  }

  async dispatch(opts: DispatchOpts): Promise<void> {
    if (await this.isLive()) {
      return;
    }

    // Resolve and apply the matching egress policy BEFORE starting the
    // container so it's in place from the very first outbound request.
    // Shared with the Isolate Sandbox via resolveSessionPolicy — both paths
    // do `applyEgressPolicy(req, env, { policy })` at runtime, so any rule
    // (allow/deny/header-injection/VPC-route/proxy fn) behaves the same
    // for either backend.
    try {
      const compiled = await resolveSessionPolicy(this.env, opts.sessionId);
      if (compiled) await this.applyPolicy(compiled);
    } catch (error) {
      console.warn(
        `[sandbox] failed to apply egress policy for ${opts.sessionId}: ${toErrorMessage(error)}`,
      );
    }

    // Container-side env vars. The container itself doesn't run a
    // worker process any more (see the class-level comment), but bash
    // tool calls invoked through `stock-tools.ts` inherit these via
    // `setEnvVars`, so user-authored shell snippets that consult
    // `$ANTHROPIC_SESSION_ID` still work. We deliberately do NOT
    // forward `ANTHROPIC_API_KEY` — that would leak an
    // organization-scoped credential into every bash tool call.
    const envVars: Record<string, string> = {
      ANTHROPIC_SESSION_ID: opts.sessionId,
      ANTHROPIC_ENVIRONMENT_KEY: this.env.ANTHROPIC_ENVIRONMENT_KEY,
      ANTHROPIC_WORK_ID: opts.workId,
      ANTHROPIC_ENVIRONMENT_ID: opts.environmentId,
      ANTHROPIC_BASE_URL: opts.baseURL,
    };

    console.log(
      `[sandbox] dispatch session=${opts.sessionId} work=${opts.workId} envKeys=${Object.keys(envVars).join(",")}`,
    );

    // Boot the sandbox runtime container AND restore the most-recent
    // /workspace snapshot before returning. ensureStarted blocks until
    // restore is complete so the dispatcher can't read or write
    // /workspace mid-restore. Shared with the PTY upgrade and /exec
    // paths.
    await this.ensureStarted(envVars);

    try {
      await this.setEnvVars(envVars);
    } catch (error) {
      console.warn(
        `[sandbox] setEnvVars failed for ${opts.sessionId}: ${toErrorMessage(error)}`,
      );
    }

    // Start the dispatcher + heartbeat in this DO. Awaited only for
    // typecheck cleanliness — both loops run detached via
    // `ctx.waitUntil` inside.
    await this.startDispatcher(opts);
  }

  // Spin up the per-session dispatcher + heartbeat loop. Safe to call
  // multiple times: an already-running dispatcher is aborted before
  // the new one starts so a redeploy that changes the tool catalog
  // takes effect on the next webhook (mirrors `IsolateRunner.start`'s
  // drift logic, without the drift comparison — cheap enough to
  // restart unconditionally).
  //
  // Two loops run in parallel under the same controller:
  //   - `runCustomToolDispatcher` reads the session event stream and
  //     answers both stock and custom tool calls.
  //   - `runHeartbeatLoop` keeps the platform's work-item lease alive
  //     and tears the controller down when the platform signals
  //     `state: stopping` (or a 4xx response means we lost the lease).
  // When either loop exits we force-stop the work item so the lease
  // releases cleanly — the equivalent of `EnvironmentWorker.handleItem()`
  // returning in the SDK-recommended flow.
  private async startDispatcher(opts: DispatchOpts): Promise<void> {
    if (!this.env.ANTHROPIC_ENVIRONMENT_KEY) {
      console.warn(
        `[sandbox] ANTHROPIC_ENVIRONMENT_KEY not set — dispatcher will not start session=${opts.sessionId}`,
      );
      return;
    }

    this.customDispatchCtrl?.abort();
    const ctrl = new AbortController();
    this.customDispatchCtrl = ctrl;

    const tools = buildCfTools({
      env: this.env,
      sessionId: opts.sessionId,
      // `workspace` is intentionally omitted — MicroVM sessions don't
      // have a DO-local workspace. screenshot / image_generate detect
      // that and return image content blocks inline.
    });

    // Browser CDP tools (browser_search / browser_execute). The factory
    // spins up a Worker isolate via the parent Worker's LOADER binding
    // and calls BROWSER directly, so the container is uninvolved.
    // Gated on the bindings being present.
    if (this.env.LOADER && this.env.BROWSER) {
      try {
        tools.push(
          ...(await isolateBrowserTools({
            loader: this.env.LOADER,
            browser: this.env.BROWSER,
          })),
        );
      } catch (error) {
        console.warn(
          `[sandbox] failed to register browser tools session=${opts.sessionId}: ${toErrorMessage(error)}`,
        );
      }
    }

    // Stock toolset handlers (bash/read/write/edit/glob/grep) that
    // delegate to the container via the Sandbox SDK. This is what
    // makes the DO a complete replacement for `ant beta:worker run` —
    // every `agent.tool_use` event now has a worker-side answer.
    const stockTools = buildMicrovmStockTools(this);

    // Under the 0.96 SDK the environment key authenticates the work
    // queue, the event stream, the heartbeat endpoint, and the
    // force-stop endpoint. `apiKey: null` avoids the SDK's process.env
    // backfill, which would send both bearer + x-api-key and trip the
    // managed-agents 401. (Same rationale as `bearerClient` in
    // src/webhooks.ts.)
    const client = new Anthropic({
      apiKey: null,
      authToken: this.env.ANTHROPIC_ENVIRONMENT_KEY,
      baseURL: opts.baseURL,
    });

    console.log(
      `[sandbox] dispatcher starting session=${opts.sessionId} work=${opts.workId} custom=${tools
        .map((t) => t.name)
        .join(",")} stock=${stockTools.map((t) => t.name).join(",")}`,
    );

    // Detached run. The DO stays alive while either loop is in
    // flight; `ctrl.signal` aborts both together. `Promise.allSettled`
    // means a heartbeat failure (which calls `ctrl.abort()`) lets the
    // dispatcher drain instead of leaving it dangling.
    //
    // `onInFlightChange` fires every time a tool starts or finishes.
    // While the count is > 0 we bump the container's activity timer
    // so a long-running `bash` (npm install, test suite) can't trigger
    // the SESSION_IDLE_TTL reaper mid-call.
    const { workId, environmentId, sessionId } = opts;
    this.ctx.waitUntil(
      (async () => {
        try {
          await Promise.allSettled([
            runCustomToolDispatcher({
              client,
              sessionId,
              tools,
              stockTools,
              signal: ctrl.signal,
              onInFlightChange: (count) => {
                if (count > 0) {
                  void this.renewContainerActivity().catch(() => {
                    // Best-effort.
                  });
                }
              },
            }).catch((error) => {
              console.error(
                `[sandbox] dispatcher loop failed session=${sessionId}: ${toErrorMessage(error)}`,
              );
            }),
            runHeartbeatLoop({
              client,
              workId,
              environmentId,
              signal: ctrl.signal,
              abort: () => ctrl.abort(),
              logPrefix: "[sandbox]",
            }).catch((error) => {
              console.error(
                `[sandbox] heartbeat loop failed session=${sessionId} work=${workId}: ${toErrorMessage(error)}`,
              );
            }),
          ]);
        } finally {
          if (this.customDispatchCtrl === ctrl) {
            this.customDispatchCtrl = undefined;
          }
          // Force-stop the work item so the lease releases cleanly,
          // regardless of why the loops exited. Best-effort: a 4xx
          // here (item already stopped, etc.) is logged and swallowed.
          // Equivalent of `EnvironmentWorker.handleItem()` returning.
          try {
            await client.beta.environments.work.stop(workId, {
              environment_id: environmentId,
              force: true,
              betas: [ANTHROPIC_BETA],
            });
          } catch (error) {
            console.warn(
              `[sandbox] force-stop failed session=${sessionId} work=${workId}: ${toErrorMessage(error)}`,
            );
          }
          console.log(`[sandbox] dispatcher exited session=${sessionId}`);
        }
      })(),
    );
  }
}

export function getSessionSandbox(env: Env, sessionId: string) {
  return getSandbox(env.Sandbox, sessionId, {
    sleepAfter: SESSION_IDLE_TTL,
  });
}

export async function getContainerStatus(
  sessionId: string,
  env: Env,
): Promise<string> {
  try {
    const sandbox = getSessionSandbox(env, sessionId);
    const state = await sandbox.getState();

    if (state.status === "healthy") {
      return "running";
    }

    if (state.status === "stopped_with_code") {
      return "stopped";
    }

    return state.status;
  } catch (error) {
    console.warn(
      `[status] failed to read container state for ${sessionId}: ${toErrorMessage(error)}`,
    );
    return "unknown";
  }
}
