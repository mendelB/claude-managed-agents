import type Anthropic from "@anthropic-ai/sdk";
import type { BetaRunnableTool } from "@anthropic-ai/sdk/lib/tools/BetaRunnableTool";
import { ANTHROPIC_BETA } from "../anthropic";

// Why this file exists
// --------------------
// MicroVM-side dispatcher for `agent.custom_tool_use` events. The
// `Sandbox` DO (src/microvm/sandbox.ts) imports this to answer cf_*
// and user-defined custom tool calls; the container's
// `ant beta:worker run` process owns the stock toolset
// (`agent.tool_use`) but has no Worker-binding access, so the DO has
// to answer customs out-of-band.
//
// Not used by the Isolate runner any more. Under the 0.96 SDK,
// `SessionToolRunner` (`client.beta.sessions.events.toolRunner`)
// dispatches BOTH `agent.tool_use` AND `agent.custom_tool_use`
// events AND reconciles past events on every (re)connect, so the
// Isolate path runs that alone. Keeping both was an attractive
// nuisance: when the parallel dispatcher's view of `tools` was
// momentarily incomplete it would post `tool "<name>" not implemented`
// and beat the SDK runner's correct answer to the punch.
//
// What this module gives MicroVM that the SDK runner doesn't:
//  1. Reconcile-across-disconnect: we walk session events on every
//     stream reconnect and pre-answer custom_tool_use events that
//     arrived while the DO was asleep (eviction, deploy, crash).
//  2. Workers-DO-aware error handling: per-tool timeout, content-block
//     formatting matching the SDK runner, and integration with a shared
//     AbortController.
//
// Heartbeat and lifecycle ownership live in the container — this
// module never touches the work item itself.

const TOOL_TIMEOUT_MS = 120_000;
const STREAM_BACKOFF_START_MS = 500;
const STREAM_BACKOFF_CAP_MS = 10_000;
const SEND_RETRIES = 3;

// Subset of the SDK's BetaManagedAgentsAgentCustomToolUseEvent shape we
// actually consume. Keeping a local interface avoids importing from deep
// SDK paths just to reach the type.
interface CustomToolUseEvent {
  type: "agent.custom_tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

interface UserCustomToolResultEvent {
  type: "user.custom_tool_result";
  custom_tool_use_id: string;
}

interface UnknownEvent {
  type: string;
}

type SessionEvent = CustomToolUseEvent | UserCustomToolResultEvent | UnknownEvent;

export interface CustomDispatchOpts {
  client: Anthropic;
  sessionId: string;
  tools: BetaRunnableTool[];
  // Shared with the SDK SessionToolRunner's controller — aborting either side
  // brings the whole runner down together. Required, not optional, so callers
  // don't accidentally start a dispatcher with no shutdown path.
  signal: AbortSignal;
}

// Run a parallel dispatcher for `agent.custom_tool_use` events. Returns when
// `signal` aborts or a fatal stream error occurs. Caught errors are logged
// and the stream reconnects with exponential backoff.
export async function runCustomToolDispatcher(opts: CustomDispatchOpts): Promise<void> {
  const { client, sessionId, tools, signal } = opts;
  const toolByName = new Map(tools.map((t) => [t.name, t]));
  const seen = new Set<string>();
  const answered = new Set<string>();

  // Reconcile first so any custom_tool_use events that arrived before the
  // DO booted (or during a reconnect gap on the SDK side) get answered.
  await reconcile({ client, sessionId, signal, toolByName, seen, answered });

  let backoff = STREAM_BACKOFF_START_MS;
  while (!signal.aborted) {
    try {
      const stream = await client.beta.sessions.events.stream(
        sessionId,
        { betas: [ANTHROPIC_BETA] },
        { signal },
      );
      for await (const ev of stream as AsyncIterable<SessionEvent>) {
        if (signal.aborted) return;
        backoff = STREAM_BACKOFF_START_MS;
        await handleEvent({ client, sessionId, signal, toolByName, seen, answered }, ev);
      }
    } catch (error) {
      if (signal.aborted) return;
      console.warn(
        `[isolate][custom-dispatch] stream disconnected session=${sessionId}, reconnecting in ${backoff}ms: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
    if (signal.aborted) return;
    // Reconcile across the disconnect window so we don't lose any events
    // emitted while we were detached.
    await reconcile({ client, sessionId, signal, toolByName, seen, answered });
    await sleep(backoff, signal);
    backoff = Math.min(backoff * 2, STREAM_BACKOFF_CAP_MS);
  }
}

interface Ctx {
  client: Anthropic;
  sessionId: string;
  signal: AbortSignal;
  toolByName: Map<string, BetaRunnableTool>;
  seen: Set<string>;
  answered: Set<string>;
}

async function reconcile(ctx: Ctx): Promise<void> {
  const pending: CustomToolUseEvent[] = [];
  try {
    for await (const ev of ctx.client.beta.sessions.events.list(
      ctx.sessionId,
      { limit: 1000, betas: [ANTHROPIC_BETA] },
      { signal: ctx.signal },
    ) as AsyncIterable<SessionEvent>) {
      if (ev.type === "agent.custom_tool_use") {
        const e = ev as CustomToolUseEvent;
        if (!ctx.seen.has(e.id)) {
          ctx.seen.add(e.id);
          pending.push(e);
        }
      } else if (ev.type === "user.custom_tool_result") {
        ctx.answered.add((ev as UserCustomToolResultEvent).custom_tool_use_id);
      }
    }
  } catch (error) {
    if (!ctx.signal.aborted) {
      console.warn(
        `[isolate][custom-dispatch] reconcile failed session=${ctx.sessionId}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
    return;
  }
  for (const ev of pending) {
    if (ctx.signal.aborted) return;
    if (ctx.answered.has(ev.id)) continue;
    await execute(ctx, ev);
  }
}

async function handleEvent(ctx: Ctx, ev: SessionEvent): Promise<void> {
  if (ev.type === "agent.custom_tool_use") {
    const e = ev as CustomToolUseEvent;
    if (ctx.seen.has(e.id)) return;
    ctx.seen.add(e.id);
    await execute(ctx, e);
  } else if (ev.type === "user.custom_tool_result") {
    ctx.answered.add((ev as UserCustomToolResultEvent).custom_tool_use_id);
  }
  // All other event types — agent.tool_use, agent.message, idle, etc — are
  // either someone else's job (SDK dispatcher) or not actionable here.
}

async function execute(ctx: Ctx, ev: CustomToolUseEvent): Promise<void> {
  console.log(
    `[isolate][custom-dispatch] tool=${ev.name} start session=${ctx.sessionId} use_id=${ev.id}`,
  );
  const start = Date.now();
  const tool = ctx.toolByName.get(ev.name);

  let content: string | unknown[];
  let isError: boolean;

  if (!tool) {
    // Not ours — skip, do NOT post a result. This dispatcher is no longer the
    // sole fulfiller of custom_tool_use on a session: the container `ant`
    // runner answers them too, and for some agents an EXTERNAL orchestrator
    // owns custom tools this dispatcher has never heard of (e.g. MendelBot's
    // L3 reviewer fulfilling its `ctx_*` context tools out-of-band over the
    // session event stream). Posting a "not found" result for a tool we don't
    // own races that fulfiller and poisons the model with a spurious error
    // alongside the real answer — see
    // notes/2026-06-08-l3-custom-tool-triple-fulfillment-bug.md. Answer ONLY
    // tools in our registry; leave the rest for whoever owns them. A genuinely
    // hallucinated tool with no fulfiller anywhere still gets the container
    // runner's own error, and the L1 watchdog is the final backstop.
    const known = [...ctx.toolByName.keys()].sort().join(",");
    console.log(
      `[isolate][custom-dispatch] tool=${ev.name} not ours — skipping (registered=${known || "(none)"}) session=${ctx.sessionId} use_id=${ev.id}`,
    );
    return;
  }
  {
    const toolCtrl = new AbortController();
    const onParentAbort = () => toolCtrl.abort();
    ctx.signal.addEventListener("abort", onParentAbort, { once: true });
    const timer = setTimeout(() => toolCtrl.abort(), TOOL_TIMEOUT_MS);
    try {
      const input = tool.parse ? tool.parse(ev.input) : ev.input;
      // `BetaToolRunContext.toolUse` is the new field name; the SDK
      // keeps `toolUseBlock` as a deprecated alias. None of our handlers
      // actually read it (they only use `signal`), so we just pass the
      // raw custom_tool_use event through.
      content = await tool.run(input, {
        toolUse: ev,
        signal: toolCtrl.signal,
      } as unknown as Parameters<NonNullable<BetaRunnableTool["run"]>>[1]);
      isError = false;
    } catch (err) {
      content = `Error: ${err instanceof Error ? err.message : String(err)}`;
      isError = true;
    } finally {
      clearTimeout(timer);
      ctx.signal.removeEventListener("abort", onParentAbort);
    }
  }

  const ms = Date.now() - start;
  const sample =
    typeof content === "string"
      ? content.length > 80
        ? `${content.slice(0, 80)}…`
        : content
      : "(blocks)";
  console.log(
    `[isolate][custom-dispatch] tool=${ev.name} done  session=${ctx.sessionId} ms=${ms} error=${isError} result=${JSON.stringify(sample)}`,
  );

  await postResult(ctx, ev.id, content, isError);
}

async function postResult(
  ctx: Ctx,
  customToolUseId: string,
  content: string | unknown[],
  isError: boolean,
): Promise<boolean> {
  const blocks = toContentBlocks(content);
  let lastErr: unknown;
  for (let i = 0; i < SEND_RETRIES; i++) {
    if (ctx.signal.aborted) return false;
    try {
      // The SDK's typed `events.send` only accepts a fixed param union; cast
      // through `unknown` so we can post the custom_tool_result shape. This
      // is the same escape hatch the audit module uses for the legacy
      // tool_result path — see src/isolate/audit.ts.
      await ctx.client.beta.sessions.events.send(
        ctx.sessionId,
        {
          betas: [ANTHROPIC_BETA],
          events: [
            {
              type: "user.custom_tool_result",
              custom_tool_use_id: customToolUseId,
              is_error: isError,
              content: blocks,
            },
          ],
        } as unknown as Parameters<typeof ctx.client.beta.sessions.events.send>[1],
        { signal: ctx.signal },
      );
      ctx.answered.add(customToolUseId);
      return true;
    } catch (err) {
      lastErr = err;
      const status = (err as { status?: number })?.status;
      // 4xx except 408/429 is fatal — same retry policy the SDK's sendResult uses.
      if (
        typeof status === "number" &&
        status >= 400 &&
        status < 500 &&
        status !== 408 &&
        status !== 429
      ) {
        break;
      }
      await sleep((i + 1) * 1000, ctx.signal);
    }
  }
  console.error(
    `[isolate][custom-dispatch] failed to send tool result session=${ctx.sessionId} use_id=${customToolUseId}: ${String(
      lastErr,
    )}`,
  );
  return false;
}

// Mirror the SDK dispatcher's toSessionContent — wrap a raw string into a
// text block, pass through structured blocks as-is, ensure we never post an
// empty content array (the API rejects it).
function toContentBlocks(content: string | unknown[]): unknown[] {
  if (typeof content === "string") {
    return [{ type: "text", text: content || "(no output)" }];
  }
  const out = content.map((b) => {
    const blk = b as { type?: string; text?: string };
    if (blk.type === "text") {
      return { type: "text", text: blk.text || "(no output)" };
    }
    if (blk.type === "image" || blk.type === "document") return b;
    return { type: "text", text: JSON.stringify(b) };
  });
  return out.length > 0 ? out : [{ type: "text", text: "(no output)" }];
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve();
    const onAbort = () => {
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}
