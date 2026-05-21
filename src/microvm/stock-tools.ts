import type { BetaRunnableTool } from "@anthropic-ai/sdk/lib/tools/BetaRunnableTool";
import type { Sandbox as SandboxBase } from "@cloudflare/sandbox";

// Stock toolset handlers for the MicroVM backend, registered with the
// dispatcher in `src/isolate/custom-dispatch.ts` so the worker can
// answer `agent.tool_use` events directly via the Sandbox SDK.
//
// Self-hosted Anthropic environments don't auto-execute the
// `agent_toolset_20260401` tools (bash / read / write / edit / glob /
// grep) on the platform side — every call surfaces as `agent.tool_use`
// and must be answered with `user.tool_result`, otherwise the session
// hangs on `session.status_idle.stop_reason.requires_action`.
// Previously we relied on `ant beta:worker run` inside the container
// to do that via the work-queue path; that's the part that broke.
//
// Tool names + JSON-Schema shapes here mirror the SDK's
// `betaAgentToolset20260401` so the model (which sees the SDK's own
// schema via `transformAgentToolsForBackend`'s `agent_toolset_20260401`
// passthrough) keeps working unchanged. Our `description` and
// `input_schema` are NEVER sent on the wire — they exist only to
// satisfy `BetaRunnableTool`'s type contract.

type SandboxLike = Pick<
  SandboxBase<unknown>,
  "exec" | "readFile" | "writeFile" | "mkdir"
>;

// Match the `--workdir /workspace` flag on `ant beta:worker run` so a
// model that emits a relative `file_path` lands in the same place its
// `bash` commands would see. Absolute paths are passed through unchanged.
const WORKDIR = "/workspace";

// The dispatcher's per-tool wall-clock is 120s. We give bash 60s by
// default so a two-step turn (bash → read → bash) still fits, and cap
// at 110s so a tool that hits the cap still leaves the dispatcher
// headroom to send the result.
const BASH_DEFAULT_TIMEOUT_MS = 60_000;
const BASH_MAX_TIMEOUT_MS = 110_000;

// Cap on bytes returned to the model. `read` gets a generous 5 MiB to
// match the SDK; everything else (bash/glob/grep) caps tighter so a
// runaway result doesn't blow out the model's context.
const READ_MAX_BYTES = 5 * 1024 * 1024;
const COMMAND_OUTPUT_MAX_BYTES = 64 * 1024;

function resolvePath(p: unknown): string {
  if (typeof p !== "string" || p.length === 0) {
    throw new Error("file_path is required");
  }
  if (p.startsWith("/")) return p;
  return `${WORKDIR}/${p.replace(/^\.\//, "")}`;
}

function shellQuote(s: string): string {
  // Single-quote and escape any embedded single quotes by closing the
  // quote, escaping, and reopening — POSIX-safe across the shells in
  // any Sandbox container image.
  return `'${s.replace(/'/g, "'\\''")}'`;
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, max)}\n…(truncated, ${s.length - max} bytes more)`;
}

// `BetaRunnableTool`'s type contract requires `description` and
// `input_schema` even though we never send them on the wire — the
// agent definition ships the SDK's own toolset descriptor. This helper
// stamps out the matching shape with minimum boilerplate and keeps the
// `as unknown as BetaRunnableTool` cast in one place.
function tool<T>(
  name: string,
  run: (args: T, ctx?: { signal?: AbortSignal }) => Promise<string>,
): BetaRunnableTool {
  return {
    name,
    description: name,
    input_schema: { type: "object", properties: {} },
    parse: (input: unknown) => input as T,
    run,
  } as unknown as BetaRunnableTool;
}

function bashTool(sandbox: SandboxLike): BetaRunnableTool {
  // The SDK ships a persistent shell session. Sandbox SDK `exec()` is
  // one-shot, so cwd and env-var mutations don't survive between
  // calls; the model handles this by chaining (`cd dir && cmd`). The
  // alternative — a long-lived bash via `startProcess` with stdin
  // piping — is materially more code than this is worth.
  return tool<{ command?: string; timeout_ms?: number }>(
    "bash",
    async (args, ctx) => {
      const command = args.command;
      if (typeof command !== "string" || command.length === 0) {
        throw new Error("bash: command is required");
      }
      const timeout = Math.min(
        Math.max(args.timeout_ms ?? BASH_DEFAULT_TIMEOUT_MS, 1),
        BASH_MAX_TIMEOUT_MS,
      );
      const r = await sandbox.exec(command, {
        cwd: WORKDIR,
        timeout,
        signal: ctx?.signal,
        origin: "user",
      });
      const stdout = r.stdout ?? "";
      const stderr = r.stderr ?? "";
      const merged = stderr ? `${stdout}\n[stderr]\n${stderr}` : stdout;
      const out = truncate(merged.trimEnd(), COMMAND_OUTPUT_MAX_BYTES);
      // Surface non-zero exit so the model can react instead of
      // treating empty stdout on a failed command as a silent success.
      if ((r.exitCode ?? 0) !== 0) {
        throw new Error(`bash: exit ${r.exitCode}\n${out || "(no output)"}`);
      }
      return out || "(no output)";
    },
  );
}

function readTool(sandbox: SandboxLike): BetaRunnableTool {
  return tool<{ file_path?: string; view_range?: number[] }>(
    "read",
    async (args) => {
      const abs = resolvePath(args.file_path);
      const r = await sandbox.readFile(abs);
      if (!r.success) throw new Error(`read: failed to read ${abs}`);
      let content = r.content ?? "";
      const [start, end] = args.view_range ?? [];
      if (typeof start === "number" && typeof end === "number" && start >= 1 && end >= start) {
        content = content.split("\n").slice(start - 1, end).join("\n");
      }
      return truncate(content, READ_MAX_BYTES);
    },
  );
}

function writeTool(sandbox: SandboxLike): BetaRunnableTool {
  return tool<{ file_path?: string; content?: string }>(
    "write",
    async (args) => {
      const abs = resolvePath(args.file_path);
      const body = args.content ?? "";
      // Parent dirs may not exist yet — `mkdir(..., { recursive: true })`
      // is idempotent so we always call it. Cheaper than a stat-then-mkdir.
      const dir = abs.replace(/\/[^/]*$/, "");
      if (dir && dir !== abs) {
        await sandbox.mkdir(dir, { recursive: true });
      }
      const r = await sandbox.writeFile(abs, body);
      if (!r.success) throw new Error(`write: failed to write ${abs}`);
      return `wrote ${new TextEncoder().encode(body).length} bytes to ${abs}`;
    },
  );
}

function editTool(sandbox: SandboxLike): BetaRunnableTool {
  return tool<{
    file_path?: string;
    old_string?: string;
    new_string?: string;
    replace_all?: boolean;
  }>("edit", async (args) => {
    const { old_string, new_string, replace_all } = args;
    if (typeof old_string !== "string" || old_string.length === 0) {
      throw new Error("edit: old_string is required");
    }
    const abs = resolvePath(args.file_path);
    const replacement = new_string ?? "";
    const r = await sandbox.readFile(abs);
    if (!r.success) throw new Error(`edit: failed to read ${abs}`);
    const original = r.content ?? "";
    const occurrences = original.split(old_string).length - 1;
    if (occurrences === 0) {
      throw new Error(`edit: old_string not found in ${abs}`);
    }
    if (occurrences > 1 && !replace_all) {
      throw new Error(
        `edit: old_string matches ${occurrences} times in ${abs}; pass replace_all=true to replace all`,
      );
    }
    const updated = replace_all
      ? original.replaceAll(old_string, replacement)
      : original.replace(old_string, replacement);
    const w = await sandbox.writeFile(abs, updated);
    if (!w.success) throw new Error(`edit: failed to write ${abs}`);
    return `edited ${abs} (${occurrences} replacement${occurrences === 1 ? "" : "s"})`;
  });
}

function globTool(sandbox: SandboxLike): BetaRunnableTool {
  return tool<{ pattern?: string; path?: string }>("glob", async (args) => {
    const { pattern, path } = args;
    if (typeof pattern !== "string" || pattern.length === 0) {
      throw new Error("glob: pattern is required");
    }
    const root = path ? resolvePath(path) : WORKDIR;
    // Strip a leading `**/` because `find -name X` already recurses;
    // for simple `*.ts` patterns this is a no-op.
    const namePattern = pattern.replace(/^\*\*\//, "");
    const cmd = `find ${shellQuote(root)} -type f -name ${shellQuote(namePattern)} 2>/dev/null | head -1000`;
    const r = await sandbox.exec(cmd, {
      cwd: WORKDIR,
      timeout: 15_000,
      origin: "internal",
    });
    return truncate((r.stdout ?? "").trim() || "(no matches)", COMMAND_OUTPUT_MAX_BYTES);
  });
}

function grepTool(sandbox: SandboxLike): BetaRunnableTool {
  // Prefer ripgrep (faster, sane defaults); fall back to `grep -rn` on
  // exit 127 ("command not found") so we work on container images that
  // don't ship rg.
  return tool<{ pattern?: string; path?: string; glob?: string; type?: string }>(
    "grep",
    async (args) => {
      const { pattern, path, glob, type: fileType } = args;
      if (typeof pattern !== "string" || pattern.length === 0) {
        throw new Error("grep: pattern is required");
      }
      const root = path ? resolvePath(path) : WORKDIR;
      const flags = ["-n", "--no-heading"];
      if (fileType) flags.push("-t", fileType);
      if (glob) flags.push("-g", glob);
      const rgCmd = `rg ${flags.join(" ")} ${shellQuote(pattern)} ${shellQuote(root)} 2>/dev/null | head -500`;
      let r = await sandbox.exec(rgCmd, { cwd: WORKDIR, timeout: 30_000, origin: "internal" });
      if ((r.exitCode ?? 0) === 127) {
        const grepCmd = `grep -rn ${shellQuote(pattern)} ${shellQuote(root)} 2>/dev/null | head -500`;
        r = await sandbox.exec(grepCmd, { cwd: WORKDIR, timeout: 30_000, origin: "internal" });
      }
      return truncate((r.stdout ?? "").trim() || "(no matches)", COMMAND_OUTPUT_MAX_BYTES);
    },
  );
}

// The full stock toolset. Order is stable so logs comparing the
// registry against a snapshot don't flap. Tools are independent — no
// shared state.
export function buildMicrovmStockTools(sandbox: SandboxLike): BetaRunnableTool[] {
  return [
    bashTool(sandbox),
    readTool(sandbox),
    writeTool(sandbox),
    editTool(sandbox),
    globTool(sandbox),
    grepTool(sandbox),
  ];
}
