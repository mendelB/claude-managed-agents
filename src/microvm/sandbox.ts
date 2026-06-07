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

// One Sandbox per session. The image entrypoint is `ant beta:worker run …`,
// so the container runs its own work loop once started — we just hand it the
// ANTHROPIC_* env vars from the polled work item via `start({ envVars })`.
//
// Two dispatchers run against each session, on disjoint event types:
//
//   1. `ant beta:worker run` inside the container — handles `agent.tool_use`
//      events emitted by Anthropic's stock toolset (bash / read / write /
//      edit / glob / grep / web_fetch / web_search). Renamed from
//      `ant worker dispatch` in the 0.96 SDK / ant 1.8 release.
//   2. `runCustomToolDispatcher` running here in the DO — handles
//      `agent.custom_tool_use` events for the cf_* family and any
//      user-defined tools from `src/tools/custom-tools.ts`. The handlers
//      close over the Worker's `env`, so tools have direct access to
//      KV, R2, D1, AI, VPC services, Email Routing, etc., without the
//      container needing a network round-trip back to the Worker.
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

    // Build the env var bag for the container.
    //
    // Under the 0.96 SDK / ant 1.8 release the environment key is the
    // single credential for the whole worker flow — `ant beta:worker run`
    // uses it to authenticate poll/ack/heartbeat/force-stop AND the
    // session event stream AND skill download. There is no separate
    // per-work-item session token any more. The CLI's run env contract
    // is `ANTHROPIC_{SESSION_ID,ENVIRONMENT_KEY,WORK_ID,ENVIRONMENT_ID,BASE_URL}`.
    //
    // We deliberately do NOT forward `ANTHROPIC_API_KEY` here. The Anthropic
    // SDK's `readEnv('ANTHROPIC_API_KEY')` fallback auto-fills `apiKey`
    // when the constructor receives only `authToken`, and the
    // managed-agents server rejects requests that carry both
    // `Authorization: Bearer` and `x-api-key` with 401. With both vars in
    // the container's `process.env`, `ant beta:worker run` ends up
    // sending the rejected combo and every per-session call 401s.
    // Anything user-authored that needs an API key should plumb it
    // through a differently-named env var.
    const envVars: Record<string, string> = {
      ANTHROPIC_SESSION_ID: opts.sessionId,
      ANTHROPIC_ENVIRONMENT_KEY: this.env.ANTHROPIC_ENVIRONMENT_KEY,
      ANTHROPIC_WORK_ID: opts.workId,
      ANTHROPIC_ENVIRONMENT_ID: opts.environmentId,
      ANTHROPIC_BASE_URL: opts.baseURL,
    };

    // Forward GITHUB_TOKEN if the control plane has one. Self-hosted
    // environments don't support the platform's `resources[]` repo mount
    // (POST /v1/sessions returns 400 on resources), so the L3 agent has
    // to clone its own repos from inside the sandbox. The token has to
    // reach the container somehow — env-var forwarding from the control
    // plane is the cleanest channel (cleaner than embedding in the task
    // prompt, where it'd be stored in the session event stream).
    if (this.env.GITHUB_TOKEN) {
      envVars.GITHUB_TOKEN = this.env.GITHUB_TOKEN;
    }

    // git on Debian ships linked against libcurl3-gnutls, which treats the
    // Cloudflare egress proxy's abnormal TLS close (curl error 56 'unexpected
    // eof while reading') as a hard handshake failure even when the CA is
    // trusted and the HTTP response was actually received. OpenSSL-linked
    // tools (curl, openssl s_client) just warn and accept the response; git
    // aborts. GIT_SSL_NO_VERIFY at the env-var level skips git's TLS
    // verification entirely. Safe because the egress proxy IS the trust
    // boundary — all outbound traffic is intercepted, inspected, and
    // policy-checked at the Cloudflare network layer before reaching the
    // public internet. The container has no other path out.
    // Diagnosed 2026-06-07 — first dispatch after fixing GITHUB_TOKEN failed
    // on `fatal: GnuTLS recv error (-110): The TLS connection was non-
    // properly terminated.`
    envVars.GIT_SSL_NO_VERIFY = "true";

    console.log(
      `[sandbox] dispatch session=${opts.sessionId} work=${opts.workId} envKeys=${Object.keys(envVars).join(",")}`,
    );

    // Boot the sandbox runtime container (port 3000) AND restore the
    // most-recent /workspace snapshot before returning. ensureStarted
    // blocks until the restore attempt is complete so the runner can't
    // read or write /workspace mid-restore. Shared with the PTY
    // upgrade and /exec paths so every "turn the sandbox on" entry
    // point gets the same restore-first guarantee.
    //
    // setEnvVars below handles container-wide env propagation for
    // subsequent exec/terminal calls; the envVars passed here populate
    // the initial container process.
    await this.ensureStarted(envVars);

    try {
      await this.setEnvVars(envVars);
    } catch (error) {
      console.warn(
        `[sandbox] setEnvVars failed for ${opts.sessionId}: ${toErrorMessage(error)}`,
      );
    }

    // Install the Cloudflare egress TLS intercept CA into the container's
    // trust store BEFORE starting the runner. The control plane's outbound
    // policy intercepts HTTPS at the network layer; the sandbox sees a cert
    // signed by "Cloudflare TLS proxy-everything Intercept CA" for every
    // outbound host. That CA is NOT in the standard Debian bundle (147 certs,
    // none of which are Cloudflare's), so Go's net/http rejects the handshake
    // with `x509: certificate signed by unknown authority`. ant beta:worker
    // run is Go-compiled, so it hits this every time and stalls — the model
    // submits agent.tool_use, the runner can't reach Anthropic to claim or
    // post results, the session sits in requires_action forever.
    //
    // Curl + openssl trust the CA somehow (BoringSSL build with embedded
    // root? legacy bundle path?), but Go's SystemCertPool only reads
    // /etc/ssl/certs/ca-certificates.crt — adding the CA there and running
    // update-ca-certificates makes Go trust it too.
    //
    // The CA can't be baked into the Dockerfile because the cert chain is
    // only visible from inside the sandbox at runtime (build happens locally
    // where api.anthropic.com hits Anthropic's real CA). Extracting via
    // openssl s_client at boot is the cleanest channel.
    //
    // Diagnosed 2026-06-07 — first real L3 dispatch sat in requires_action
    // for 60 minutes before the MendelBot watchdog timed it out.
    const caInstall = [
      "set -e",
      // Capture the live cert chain — bash heredoc + openssl s_client.
      "echo Q | openssl s_client -showcerts -connect api.anthropic.com:443 -servername api.anthropic.com 2>/dev/null > /tmp/cf_chain.txt",
      // Split into per-cert PEM files. The awk reopens output on each BEGIN.
      "awk '/-----BEGIN CERTIFICATE-----/{c++; out=\"/tmp/cf_chain_\" c \".pem\"} c{print > out}' /tmp/cf_chain.txt",
      // Find the Cloudflare-issued CA (depth=1 in the chain) and install.
      "for f in /tmp/cf_chain_*.pem; do",
      "  if openssl x509 -in \"$f\" -noout -issuer 2>/dev/null | grep -q Cloudflare; then",
      "    cat \"$f\" > /usr/local/share/ca-certificates/cf-intercept.crt",
      "  fi",
      "done",
      "update-ca-certificates >/dev/null 2>&1 || true",
      // Print result for control-plane logs.
      "echo CA-install:$(grep -c 'BEGIN CERT' /etc/ssl/certs/ca-certificates.crt) certs in bundle",
    ].join("\n");
    try {
      const caResult = await (this as any).exec(caInstall, { timeout: 10_000 });
      console.log(
        `[sandbox] CA install session=${opts.sessionId} exit=${caResult?.exitCode ?? "?"} out=${(caResult?.stdout || "").trim()}`,
      );
    } catch (error) {
      console.warn(
        `[sandbox] CA install failed for ${opts.sessionId}: ${toErrorMessage(error)} — runner will likely fail TLS to api.anthropic.com`,
      );
    }

    try {
      // `ant beta:worker run` (formerly `ant worker dispatch`) reads the
      // ANTHROPIC_* env vars we set above. `--unrestricted-paths` replaces
      // the old `--allow-absolute-paths` flag.
      const command =
        "ant beta:worker run --workdir /workspace --unrestricted-paths --max-idle 60s --log-format json";
      const proc = await this.startProcess(command, {
        env: envVars,
        cwd: "/workspace",
      });
      console.log(
        `[sandbox] runner started session=${opts.sessionId} pid=${proc.id ?? "?"} status=${proc.status ?? "?"}`,
      );
    } catch (error) {
      console.error(
        `[sandbox] failed to launch runner for ${opts.sessionId}: ${toErrorMessage(error)}`,
      );
    }

    // Kick off the custom-tool dispatcher inside this DO. It polls
    // Anthropic for `agent.custom_tool_use` events and answers them
    // with `env`-backed tool handlers — so cf_* / user-defined tools
    // never need to traverse the container's network boundary.
    // Awaited only for typecheck cleanliness — the dispatcher itself
    // runs detached via `ctx.waitUntil` inside.
    await this.startCustomToolDispatcher(opts);
  }

  // Start the parallel custom-tool dispatcher for this session. Safe to
  // call multiple times: an already-running dispatcher is aborted and
  // restarted so a redeploy that changes the tool set takes effect on
  // the next webhook (mirrors `IsolateRunner.start`'s drift logic, just
  // without the drift comparison — the dispatcher is cheap enough that
  // unconditional restart on every cold-boot is fine).
  private async startCustomToolDispatcher(opts: DispatchOpts): Promise<void> {
    if (!this.env.ANTHROPIC_ENVIRONMENT_KEY) {
      console.warn(
        `[sandbox] ANTHROPIC_ENVIRONMENT_KEY not set — custom-tool dispatcher will not start session=${opts.sessionId}`,
      );
      return;
    }

    this.customDispatchCtrl?.abort();
    const ctrl = new AbortController();
    this.customDispatchCtrl = ctrl;

    const tools = buildCfTools({
      env: this.env,
      sessionId: opts.sessionId,
      // `workspace` is intentionally omitted — Sandbox sessions have no
      // DO-local workspace. screenshot / image_generate detect
      // this and return image content blocks inline instead of writing
      // to a workspace path.
    });

    // Browser CDP tools (browser_search / browser_execute). The factory
    // spins up a Worker isolate via the parent Worker's LOADER binding
    // and calls BROWSER directly, so the container is uninvolved — the
    // same code path the Isolate backend uses works here unchanged.
    // Gated on the bindings being present so a deploy without them
    // doesn't register placeholder tools.
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

    if (tools.length === 0) {
      console.log(
        `[sandbox] no custom tools available for session=${opts.sessionId} — dispatcher not started`,
      );
      this.customDispatchCtrl = undefined;
      return;
    }

    // Under the 0.96 SDK the environment key authenticates the session
    // event stream as well as the worker flow — no separate API key
    // required for events.stream / .list / .send.
    //
    // apiKey: null avoids the SDK's process.env backfill, which would
    // send both bearer + x-api-key and trip the managed-agents 401.
    // (Same rationale as `bearerClient` in src/webhooks.ts.)
    const client = new Anthropic({
      apiKey: null,
      authToken: this.env.ANTHROPIC_ENVIRONMENT_KEY,
      baseURL: opts.baseURL,
    });

    console.log(
      `[sandbox] custom-tool dispatcher starting session=${opts.sessionId} tools=${tools
        .map((t) => t.name)
        .join(",")}`,
    );

    // Detached run. The DO stays alive while the dispatcher polls;
    // when `ctrl.signal` aborts (manual stop, activity-expired, or
    // another dispatch on the same session restarts us) the dispatcher
    // returns and we clear the controller.
    this.ctx.waitUntil(
      runCustomToolDispatcher({
        client,
        sessionId: opts.sessionId,
        tools,
        signal: ctrl.signal,
      })
        .catch((error) => {
          console.error(
            `[sandbox] custom-tool dispatcher failed session=${opts.sessionId}: ${toErrorMessage(error)}`,
          );
        })
        .finally(() => {
          if (this.customDispatchCtrl === ctrl) {
            this.customDispatchCtrl = undefined;
          }
          console.log(
            `[sandbox] custom-tool dispatcher exited session=${opts.sessionId}`,
          );
        }),
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
