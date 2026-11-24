// QuickClaude agent sidecar.
//
// Speaks JSON lines over stdio with the Tauri shell:
//   in:  {type:"hello"}                                  re-send models (UI reloaded)
//        {type:"ask", text, images?, annotations?, followUp, turn, model, effort}
//                                     annotations: [{quote, note?}] the user highlighted in replies
//        {type:"stop"} | {type:"reset"}
//        {type:"history"}                                list past sessions
//        {type:"open", id, turn}                         reopen a past session
//        {type:"handoff"}                                continue the session in a terminal
//        {type:"compact", turn}                          shrink the open chat's context (/compact)
//        {type:"choice", model, effort}                  remember the picked model and effort
//        {type:"delete", id}                             delete a past session from Claude Code
//        {type:"pin", ids, pinned}                       pin or unpin chats in history
//   out: {type:"ready"} | {type:"models", models, defaultModel, defaultEffort, chosen} | {type:"session", id}
//        {type:"delta", turn, text} | {type:"tool", turn, name} | {type:"done", turn, ...}
//        {type:"history", sessions} | {type:"deleted", id} | {type:"transcript", id, turn, messages}
//        {type:"tasks", turn, count}  background tasks of the open chat
//        {type:"wake", turn, compacting?}  the open chat has a turn running that you didn't just start
//                                     (a background task finished, or a reopened chat is mid-turn or mid-/compact)
//        {type:"activity", ids}       chats with a turn or background task running
//        {type:"notify", id, title, text, current}  a chat finished; the shell alerts you if you're away
//        {type:"compacted", turn, before, after}  context tokens around a /compact
//        {type:"context", turn, tokens}  the open chat's context size, as /context counts it
//        {type:"handoff", id} | {type:"error", turn?, message}
//
// Auth comes from the local `claude` login (your Claude subscription).
// A spare session is always open and waiting, so a new question never pays
// process start-up.

import { deleteSession, getSessionMessages, listSessions, query, startup } from "@anthropic-ai/claude-agent-sdk";
import { createInterface } from "node:readline";
import { execFile, execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const HOME = homedir();

// "full" mirrors your terminal Claude Code; "lean" is the fast, tool-less answer box.
const PROFILE = process.env.QUICKCLAUDE_PROFILE ?? "full";
const MODEL = process.env.QUICKCLAUDE_MODEL; // unset: whatever your Claude Code settings use

// Apps opened from Finder or at login get a bare PATH. Borrow the login shell's,
// so Bash commands and MCP servers find the same tools as in your terminal.
try {
  const out = execFileSync(process.env.SHELL || "/bin/zsh", ["-ilc", 'printf "__PATH__%s" "$PATH"'], {
    encoding: "utf8",
    timeout: 5000,
    stdio: ["ignore", "pipe", "ignore"],
  });
  const path = out.split("__PATH__").pop();
  if (path) process.env.PATH = path;
} catch {}

const onPath = (name) =>
  (process.env.PATH ?? "")
    .split(":")
    .map((dir) => join(dir, name))
    .find((file) => existsSync(file));

// Run your installed `claude`, so the app matches your terminal's version and updates.
// Falls back to the SDK's bundled copy (dev only; the packaged app doesn't include it).
const CLAUDE = process.env.QUICKCLAUDE_CLAUDE || onPath("claude");
// The terminal workspace manager; when it's running, ⌘⇧H opens a workspace there.
const HERDR = onPath("herdr");

// Never let an API key in the environment override the subscription login.
const env = { ...process.env, CLAUDE_AGENT_SDK_CLIENT_APP: "quickclaude/0.1.0" };
delete env.ANTHROPIC_API_KEY;

const LEAN_PROMPT = `You are QuickClaude, a fast desktop assistant summoned with a hotkey.
Answer directly and concisely; lead with the answer, then only the detail that helps.
Use Markdown when it aids reading (code blocks, short lists, tables).
Use web search only when the question needs current or niche facts.
Today's date is ${new Date().toDateString()}.`;

// QuickClaude's own folder: its chats, history and any files Claude makes stay here, apart from ~.
const WORK_DIR = process.env.QUICKCLAUDE_DIR || join(HOME, "QuickClaude");
mkdirSync(WORK_DIR, { recursive: true });

const LEAN_DIR = join(HOME, ".quickclaude");
mkdirSync(LEAN_DIR, { recursive: true });

const OPTIONS =
  PROFILE === "lean"
    ? {
        cwd: LEAN_DIR,
        settingSources: [], // no CLAUDE.md / settings / MCP: faster, predictable
        systemPrompt: LEAN_PROMPT,
        tools: ["WebSearch", "WebFetch"],
        allowedTools: ["WebSearch", "WebFetch"],
      }
    : {
        // Same as running `claude` in ~/QuickClaude: your CLAUDE.md, settings,
        // MCP servers, skills, Claude Code's own system prompt and full tool set.
        // Permissions mirror your settings (bypassPermissions), so nothing is asked.
        cwd: WORK_DIR,
        settingSources: ["user", "project", "local"],
        systemPrompt: { type: "preset", preset: "claude_code" },
        tools: { type: "preset", preset: "claude_code" },
        permissionMode: "bypassPermissions",
        allowDangerouslySkipPermissions: true,
      };
Object.assign(
  OPTIONS,
  { env, includePartialMessages: true },
  MODEL && { model: MODEL },
  CLAUDE && { pathToClaudeCodeExecutable: CLAUDE },
);

const emit = (msg) => process.stdout.write(JSON.stringify(msg) + "\n");

// --- sessions -------------------------------------------------------------

function inputQueue() {
  const items = [];
  let wake = null;
  let closed = false;
  return {
    push(content) {
      items.push({ type: "user", message: { role: "user", content }, parent_tool_use_id: null });
      wake?.();
    },
    close() {
      closed = true;
      wake?.();
    },
    async *[Symbol.asyncIterator]() {
      while (true) {
        while (items.length) yield items.shift();
        if (closed) return;
        await new Promise((r) => (wake = r));
        wake = null;
      }
    },
  };
}

// Claude Code can take over a minute to start when the Mac is busy (just logged in,
// MCP servers starting), so allow more than the SDK's 60 s.
const START_TIMEOUT = 180_000;

// A session is an open Claude Code process waiting on its input queue.
// New sessions come from a pre-started process; resumed ones start cold.
async function createSession(resume) {
  const input = inputQueue();
  const q = resume
    ? query({ prompt: input, options: { ...OPTIONS, resume } })
    : (await startup({ options: OPTIONS, initializeTimeoutMs: START_TIMEOUT })).query(input);
  return {
    input,
    q,
    id: resume ?? null,
    turn: 0,
    hadText: false,
    inTurn: false, // a turn is running
    tasks: 0, // live background tasks (shell commands, agents, monitors)
    replyDue: null, // timer: a task just finished and Claude is about to reply to it
    woke: false, // this turn started on its own, not from a question
    compacting: false, // this turn is a /compact, not an answer
    compactedFrom: 0, // context tokens before the /compact that just ran
    parked: false, // no longer on screen, kept alive until its work ends
    reply: "", // text of the current turn, for a notification
    model: "", // "" = your default
    effort: "", // "" = the model's default
    close() {
      input.close();
      q.close();
    },
  };
}

let spare = null; // Promise<session>
let current = null; // session on screen
const live = new Set(); // every running session: the current one plus parked ones
let seq = 0; // bumps on every switch, so a slow start can tell it was superseded
let models = null;

// Nobody is waiting on a spare, so a failed start retries quietly before it reports.
function warmSpare(attempt = 1) {
  const pending = createSession();
  spare = pending;
  pending.catch((e) => {
    if (spare !== pending) return; // already taken by a question, which reports its own failure
    spare = null;
    if (attempt < 3) setTimeout(() => spare || warmSpare(attempt + 1), 5000);
    else emit({ type: "error", message: `Warm-up failed: ${e.message}` });
  });
}

function takeSpare() {
  const pending = spare ?? createSession();
  spare = null;
  return pending;
}

const working = (s) => s.inTurn || s.tasks > 0 || s.replyDue !== null;

// When background work finishes, Claude Code starts a turn to report it a moment later.
// Count that gap as working, but give up if no reply comes.
function expectReply(s) {
  clearTimeout(s.replyDue);
  s.replyDue = setTimeout(() => {
    s.replyDue = null;
    if (s === current) emit({ type: "tasks", turn: s.turn, count: s.tasks });
    broadcast();
    if (s.parked && !working(s)) closeSession(s);
  }, 15000);
}

function replyStarted(s) {
  clearTimeout(s.replyDue);
  s.replyDue = null;
}

// Which chats have something running, for the lamps in the history list.
function broadcast() {
  emit({ type: "activity", ids: [...live].filter((s) => s.id && working(s)).map((s) => s.id) });
}

function closeSession(s) {
  live.delete(s);
  s.close();
  broadcast();
}

// Leaving a chat: close its process, unless it still has work running.
// Then it stays alive off screen until the work (and any reply to it) is done.
function retire(s) {
  if (!s) return;
  if (s === current) current = null;
  if (working(s)) {
    s.parked = true;
    broadcast();
  } else closeSession(s);
}

function activate(s) {
  retire(current);
  current = s;
  live.add(s);
  pump(s);
}

// Bring the session's model and effort in line with the picker before sending.
async function configure(s, model, effort) {
  if (model !== undefined && model !== s.model) {
    await s.q.setModel(model || undefined);
    s.model = model;
  }
  if (effort !== undefined && effort !== s.effort) {
    await s.q.applyFlagSettings({ effortLevel: effort || null });
    s.effort = effort;
  }
}

// Highlights and notes ride along as their own text block, which the chat view never shows
// (NOISE hides it when a chat is reopened).
function annotationBlock(annotations) {
  if (!annotations?.length) return null;
  const lines = annotations.map((a, i) => {
    const quote = `${i + 1}. "${a.quote.replace(/\s+/g, " ")}"`;
    return a.note ? `${quote}\n   Note: ${a.note}` : quote;
  });
  return {
    type: "text",
    text: `<user-annotations>\nThe user highlighted these passages in your earlier replies${
      annotations.some((a) => a.note) ? ", some with a note" : ""
    }. Take them into account in this reply; they aren't visible to the user as part of their message.\n${lines.join("\n")}\n</user-annotations>`,
  };
}

function content(text, images, annotations) {
  const note = annotationBlock(annotations);
  if (!images?.length && !note) return text;
  const blocks = (images ?? []).map((img) => ({
    type: "image",
    source: { type: "base64", media_type: img.mediaType, data: img.data },
  }));
  if (note) blocks.push(note);
  return text ? [...blocks, { type: "text", text }] : blocks;
}

async function ask({ text, images, annotations, followUp, turn, model, effort }) {
  let s = followUp ? current : null;
  if (!s) {
    const mine = ++seq;
    retire(current);
    try {
      s = await takeSpare();
    } catch (e) {
      emit({ type: "error", turn, message: e.message });
      return;
    }
    if (mine !== seq) return s.close(); // another question or session replaced this one
    activate(s);
  }
  s.turn = turn;
  s.title ??= text;
  replyStarted(s);
  s.inTurn = true;
  s.reply = "";
  broadcast();
  try {
    await configure(s, model, effort);
  } catch (e) {
    emit({ type: "error", message: `Couldn't switch model/effort: ${e.message}` });
  }
  s.input.push(content(text, images, annotations));
}

// /compact in the same session: Claude Code swaps the history for a summary and carries on.
function compact({ turn }) {
  const s = current;
  if (!s?.id) {
    emit({ type: "error", message: "Ask something first; there's no chat to compact yet." });
    return;
  }
  if (working(s)) {
    emit({ type: "error", message: "Wait for the chat to finish before compacting it." });
    return;
  }
  s.turn = turn;
  s.inTurn = true;
  s.compacting = true;
  s.reply = "";
  broadcast();
  s.input.push("/compact");
}

// The chat's context size (what /context shows), for the gauge next to the model label.
// Right after a /compact the quick estimate is stale, so count it fully and report the drop.
async function reportContext(s, compactedFrom) {
  const turn = s.turn;
  let tokens = null;
  try {
    ({ totalTokens: tokens } = await s.q.getContextUsage({ detail: compactedFrom ? "full" : "summary" }));
  } catch {}
  if (s !== current) return;
  if (compactedFrom) emit({ type: "compacted", turn, before: compactedFrom, after: tokens });
  if (tokens !== null) emit({ type: "context", turn, tokens });
}

async function pump(s) {
  const onScreen = () => s === current;
  try {
    for await (const m of s.q) {
      if (m.type === "system" && m.subtype === "init") {
        if (!s.id) {
          s.id = m.session_id;
          if (onScreen()) emit({ type: "session", id: s.id });
          broadcast();
        }
      } else if (m.type === "system" && m.subtype === "background_tasks_changed") {
        // The full set each time; ambient tasks are housekeeping, not work.
        const before = s.tasks;
        s.tasks = m.tasks.filter((t) => !t.ambient).length;
        if (s.tasks < before && !s.inTurn) expectReply(s);
        if (onScreen()) emit({ type: "tasks", turn: s.turn, count: s.tasks + (s.replyDue ? 1 : 0) });
        broadcast();
        if (s.parked && !working(s)) closeSession(s);
      } else if (m.type === "system" && m.subtype === "compact_boundary") {
        // post_tokens counts only the messages, so the size after is measured when the turn ends.
        s.compactedFrom = m.compact_metadata.pre_tokens;
      } else if (m.type === "stream_event" && !m.parent_tool_use_id) {
        const ev = m.event;
        if (ev.type === "message_start" && !s.inTurn) {
          // Claude Code woke up by itself, e.g. to report a finished background task.
          replyStarted(s);
          s.inTurn = true;
          s.woke = true;
          s.reply = "";
          if (onScreen()) emit({ type: "wake", turn: s.turn });
          broadcast();
        } else if (ev.type === "content_block_start") {
          const block = ev.content_block;
          if (block.type === "text" && s.hadText) {
            s.reply += "\n\n";
            if (onScreen()) emit({ type: "delta", turn: s.turn, text: "\n\n" });
          } else if (block.type === "tool_use" || block.type === "server_tool_use") {
            if (onScreen()) emit({ type: "tool", turn: s.turn, name: block.name });
          }
        } else if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") {
          s.hadText = true;
          s.reply += ev.delta.text;
          if (onScreen()) emit({ type: "delta", turn: s.turn, text: ev.delta.text });
        }
      } else if (m.type === "result") {
        if (onScreen()) {
          reportContext(s, s.compacting && s.compactedFrom);
          emit({
            type: "done",
            turn: s.turn,
            isError: m.is_error,
            error: m.subtype === "success" ? null : (m.errors ?? []).join("\n") || m.subtype,
            result: m.subtype === "success" ? m.result : null,
            durationMs: m.duration_ms,
            costUsd: m.total_cost_usd,
          });
        }
        const text = (s.reply || m.result || "").trim();
        const compacted = s.compacting;
        s.inTurn = false;
        s.woke = false;
        s.hadText = false;
        s.compacting = false;
        s.compactedFrom = 0;
        // Fully finished (nothing left running in the background): the shell decides whether
        // you're looking at it, and otherwise plays a sound, notifies or brings the chat forward.
        // A finished /compact has no reply to tell you about.
        if (s.id && !compacted && !working(s)) emit({ type: "notify", id: s.id, title: s.title ?? "", text, current: onScreen() });
        broadcast();
        if (s.parked && !working(s)) closeSession(s);
        // Start the next spare once this answer is out, so it never competes with it.
        if (!spare) warmSpare();
      }
    }
  } catch (e) {
    if (onScreen()) emit({ type: "error", turn: s.turn, message: e.message });
  }
  // The process ended (closed by us, or it exited).
  replyStarted(s);
  live.delete(s);
  s.inTurn = false;
  s.tasks = 0;
  broadcast();
}

// --- history --------------------------------------------------------------

// Titles of listed chats, so a reopened chat can name its terminal workspace.
const titles = new Map();

// Pinned chats: id -> when it was pinned. They head the history list, newest pin first.
const PINS_FILE = join(HOME, "Library", "Application Support", "QuickClaude", "pins.json");

function readPins() {
  try {
    return JSON.parse(readFileSync(PINS_FILE, "utf8"));
  } catch {
    return {};
  }
}

function writePins(pins) {
  try {
    mkdirSync(join(PINS_FILE, ".."), { recursive: true });
    writeFileSync(PINS_FILE, JSON.stringify(pins));
  } catch {}
}

function pin({ ids, pinned }) {
  const pins = readPins();
  const now = Date.now();
  // The first id is the one you're on; it goes to the very top.
  ids.forEach((id, i) => (pinned ? (pins[id] = now - i) : delete pins[id]));
  writePins(pins);
}

async function history() {
  try {
    const sessions = await listSessions({ dir: WORK_DIR, includeWorktrees: false });
    const pins = readPins();
    const rows = sessions
      .map((s) => ({
        id: s.sessionId,
        title: s.customTitle || s.summary || s.firstPrompt || "(untitled)",
        at: s.lastModified,
        pinned: pins[s.sessionId] ?? 0,
      }))
      .sort((a, b) => b.pinned - a.pinned || b.at - a.at)
      .slice(0, 300);
    for (const r of rows) titles.set(r.id, r.title);
    emit({ type: "history", sessions: rows });
  } catch (e) {
    emit({ type: "error", message: `Couldn't read history: ${e.message}` });
  }
}

// Removes the session's transcript from Claude Code, like deleting it there.
async function remove(id) {
  if (current?.id === id) {
    ++seq;
    current = null;
  }
  for (const s of live) if (s.id === id) closeSession(s);
  try {
    await deleteSession(id, { dir: WORK_DIR });
    titles.delete(id);
    const pins = readPins();
    if (id in pins) {
      delete pins[id];
      writePins(pins);
    }
    emit({ type: "deleted", id });
  } catch (e) {
    emit({ type: "error", message: `Couldn't delete that chat: ${e.message}` });
  }
}

// Claude Code stores harness notes as user messages; keep only what a person typed.
const NOISE = /^(<command-|<local-command-|<system-reminder>|<task-notification>|<user-annotations>|Caveat: )/;
// A compacted chat's transcript starts from this summary; the earlier messages aren't returned.
const SUMMARY = /^This session is being continued from a previous conversation/;

function toChat(messages) {
  const chat = [];
  for (const m of messages) {
    if (m.parent_tool_use_id || (m.type !== "user" && m.type !== "assistant")) continue;
    const c = m.message?.content;
    const blocks = typeof c === "string" ? [{ type: "text", text: c }] : Array.isArray(c) ? c : [];
    const text = blocks
      .filter((b) => b.type === "text" && !NOISE.test(b.text))
      .map((b) => b.text)
      .join("\n\n")
      .trim();
    const images = blocks.filter((b) => b.type === "image").length;
    if (!text && !images) continue;
    if (m.type === "user" && SUMMARY.test(text)) {
      chat.push({ role: "assistant", text: "_Earlier messages were compacted._", images: 0 });
      continue;
    }
    const prev = chat[chat.length - 1];
    if (m.type === "assistant" && prev?.role === "assistant") prev.text += `\n\n${text}`;
    else chat.push({ role: m.type, text, images });
  }
  return chat;
}

async function open({ id, turn }) {
  const mine = ++seq;
  if (current?.id === id) current = null; // reopening the chat on screen: reuse it below
  else retire(current);
  try {
    const messages = await getSessionMessages(id, { dir: WORK_DIR });
    if (mine !== seq) return;
    emit({ type: "transcript", id, turn, messages: toChat(messages) });
    // Still running off screen: bring that process back instead of starting a second one.
    const running = [...live].find((x) => x.id === id);
    if (running) {
      running.parked = false;
      running.turn = turn;
      running.woke = false;
      current = running;
      emit({ type: "tasks", turn, count: running.tasks });
      if (running.inTurn) emit({ type: "wake", turn, compacting: running.compacting });
      reportContext(running);
      return;
    }
    const s = await createSession(id);
    if (mine !== seq) return s.close();
    s.turn = turn;
    s.title = titles.get(id);
    activate(s);
    broadcast();
    reportContext(s);
  } catch (e) {
    emit({ type: "error", turn, message: `Couldn't open that chat: ${e.message}` });
  }
}

// --- terminal handoff -------------------------------------------------------

const shellQuote = (s) => `'${s.replace(/'/g, `'\\''`)}'`;

const run = (file, args) =>
  new Promise((resolve, reject) =>
    execFile(file, args, { timeout: 5000 }, (err, stdout) => (err ? reject(err) : resolve(stdout))),
  );

// A new herdr workspace in QuickClaude's folder, named after the chat, with claude typed into its shell.
// Returns false when herdr isn't installed or running, so the caller can open a plain terminal.
async function openInHerdr(command, title) {
  if (!HERDR) return false;
  try {
    const label = (title || "claude").replace(/\s+/g, " ").trim().slice(0, 40);
    const out = JSON.parse(await run(HERDR, ["workspace", "create", "--cwd", WORK_DIR, "--label", label, "--focus"]));
    const pane = out.result.root_pane.pane_id;
    await run(HERDR, ["pane", "run", pane, command]);
    // Bring the terminal that hosts herdr to the front.
    if (existsSync("/Applications/Ghostty.app")) spawn("open", ["-a", "Ghostty"], { detached: true, stdio: "ignore" }).unref();
    return true;
  } catch {
    return false;
  }
}

function openTerminal(command) {
  const shell = process.env.SHELL || "/bin/zsh";
  // Run claude in a login shell, then leave you in that shell when it exits.
  const script = `${command}; exec ${shell} -l`;
  if (existsSync("/Applications/Ghostty.app")) {
    spawn("open", ["-na", "Ghostty.app", "--args", `--working-directory=${WORK_DIR}`, "-e", shell, "-lc", script], {
      detached: true,
      stdio: "ignore",
    }).unref();
  } else {
    const cmd = `cd ${shellQuote(WORK_DIR)} && ${script}`;
    const as = `tell application "Terminal" to do script ${JSON.stringify(cmd)}\ntell application "Terminal" to activate`;
    spawn("osascript", ["-e", as], { detached: true, stdio: "ignore" }).unref();
  }
}

async function handoff() {
  const s = current;
  if (!s?.id) {
    emit({ type: "error", message: "Ask something first; there's no session to continue yet." });
    return;
  }
  if (working(s)) {
    emit({ type: "error", message: "Wait for the background work to finish; moving now would stop it." });
    return;
  }
  // Close our process first so the terminal is the only one writing to the session.
  ++seq;
  current = null;
  closeSession(s);
  await new Promise((r) => setTimeout(r, 300));
  const flags = [
    `--resume ${s.id}`,
    s.model && `--model ${shellQuote(s.model)}`,
    s.effort && `--effort ${shellQuote(s.effort)}`,
  ].filter(Boolean);
  const command = `claude ${flags.join(" ")}`;
  if (!(await openInHerdr(command, s.title))) openTerminal(command);
  emit({ type: "handoff", id: s.id });
}

// --- stdin loop -----------------------------------------------------------

createInterface({ input: process.stdin }).on("line", (line) => {
  let cmd;
  try {
    cmd = JSON.parse(line);
  } catch {
    return;
  }
  if (cmd.type === "ask") ask(cmd);
  else if (cmd.type === "stop") current?.q.interrupt().catch(() => {});
  else if (cmd.type === "reset") {
    ++seq;
    retire(current);
  } else if (cmd.type === "hello") {
    if (models) emitModels();
    broadcast();
  }
  else if (cmd.type === "history") history();
  else if (cmd.type === "open") open(cmd);
  else if (cmd.type === "handoff") handoff();
  else if (cmd.type === "compact") compact(cmd);
  else if (cmd.type === "choice") saveChoice(cmd);
  else if (cmd.type === "delete") remove(cmd.id);
  else if (cmd.type === "pin") pin(cmd);
});

process.stdin.on("end", () => {
  for (const s of live) s.close();
  spare?.then((s) => s.close(), () => {});
  setTimeout(() => process.exit(0), 200);
});

// The picker offers only the latest model of each family, at medium or high effort.
const EFFORTS = ["medium", "high"];

function userEffort() {
  try {
    const level = JSON.parse(readFileSync(join(HOME, ".claude", "settings.json"), "utf8")).effortLevel;
    return EFFORTS.includes(level) ? level : "high";
  } catch {
    return "high";
  }
}

// The last picked model and effort, kept across restarts (shared by dev and installed copies).
const CHOICE_DIR = join(HOME, "Library", "Application Support", "QuickClaude");
const CHOICE_FILE = join(CHOICE_DIR, "choice.json");

function savedChoice() {
  try {
    return JSON.parse(readFileSync(CHOICE_FILE, "utf8"));
  } catch {
    return {};
  }
}

function saveChoice({ model, effort }) {
  try {
    mkdirSync(CHOICE_DIR, { recursive: true });
    writeFileSync(CHOICE_FILE, JSON.stringify({ model, effort }));
  } catch {}
}

// Defaults are the saved choice when it's still offered, else your Claude Code settings.
function emitModels() {
  const { model, effort } = savedChoice();
  const m = models.models.find((x) => x.value === model);
  emit({
    type: "models",
    models: models.models,
    defaultModel: m ? m.value : models.defaultModel,
    defaultEffort: m?.efforts.includes(effort) ? effort : models.defaultEffort,
    chosen: existsSync(CHOICE_FILE),
  });
}

function pickerModels(all) {
  // Aliases ("opus", "sonnet", …) always point at the newest model of their family.
  const latest = all.filter((m) => m.value !== "default" && !m.value.startsWith("claude-"));
  const def = all.find((m) => m.value === "default");
  const defaultModel =
    latest.find((m) => m.resolvedModel === def?.resolvedModel)?.value ?? latest[0]?.value ?? "";
  return {
    models: latest.map((m) => ({
      value: m.value,
      name: m.displayName,
      efforts: EFFORTS.filter((e) => (m.supportedEffortLevels ?? []).includes(e)),
    })),
    defaultModel,
    defaultEffort: userEffort(),
  };
}

warmSpare();
spare?.then(
  async (s) => {
    emit({ type: "ready" });
    try {
      models = pickerModels(await s.q.supportedModels());
      emitModels();
    } catch {}
  },
  () => {},
);
