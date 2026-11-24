import { createSignal, createMemo, For, Show, onMount, onCleanup, createEffect, on, type JSX } from "solid-js";
import { createStore } from "solid-js/store";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow, LogicalSize } from "@tauri-apps/api/window";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { openUrl } from "@tauri-apps/plugin-opener";
import { marked } from "marked";
import DOMPurify from "dompurify";
import "./App.css";

const WIDTH = 720;
const COMPACT_MAX = 560;
const CHAT_HEIGHT = 640;
const MAX_IMAGE_EDGE = 2000; // px; larger pastes are scaled down before sending
const MAX_IMAGE_BYTES = 3_700_000; // keeps the base64 under the API's 5 MB image limit

type Image = { mediaType: string; data: string; url: string };

type Msg = {
  role: "user" | "assistant";
  text: string;
  images?: string[]; // data URLs of images pasted in this app
  imageCount?: number; // images in a reopened chat (not shown, only counted)
  status?: string;
  error?: string;
  meta?: string;
  marked?: string; // "2 highlights · 1 note": what went with this follow-up, unseen
};

// A highlight in a reply: character offsets in the reply's rendered text, and an optional note.
// Sent once with the next follow-up (⌘↩), then shown faded; leaving the chat drops them all.
type Mark = { id: number; msg: number; start: number; end: number; quote: string; note?: string; sent: boolean };
// What a reply paints: its marks, with the number of each note in the chat.
type Paint = Mark & { n?: number };

// Note numbers and cards are QuickClaude's own; they don't count as the reply's text.
const ANNO = ".anno";

// The reply's text from its start up to a point, skipping notes, as an offset.
function offsetIn(root: HTMLElement, node: Node, offset: number) {
  const r = document.createRange();
  r.setStart(root, 0);
  r.setEnd(node, offset);
  const frag = r.cloneContents();
  frag.querySelectorAll(ANNO).forEach((el) => el.remove());
  return frag.textContent?.length ?? 0;
}

// Wrap the reply's text between two offsets in <mark>s, one per text node it crosses.
function wrap(root: HTMLElement, start: number, end: number, make: () => HTMLElement) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement?.closest(ANNO) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const nodes: { node: Text; at: number }[] = [];
  for (let at = 0, n = walker.nextNode() as Text | null; n; n = walker.nextNode() as Text | null) {
    nodes.push({ node: n, at });
    at += n.data.length;
  }
  for (const { node, at } of nodes) {
    const a = Math.max(start - at, 0);
    const b = Math.min(end - at, node.data.length);
    if (a >= b || !node.data.slice(a, b).trim()) continue;
    if (b < node.data.length) node.splitText(b);
    const piece = a > 0 ? node.splitText(a) : node;
    const el = make();
    piece.replaceWith(el);
    el.append(piece);
  }
}

// Highlights in yellow; each note gets a number at the end of its highlight and a card
// under the paragraph (or list, table, code block) where the highlight ends.
function paint(root: HTMLElement, marks: Paint[]) {
  for (const m of [...marks].sort((a, b) => a.start - b.start)) {
    wrap(root, m.start, m.end, () => {
      const el = document.createElement("mark");
      el.className = m.sent ? "hl sent" : "hl";
      el.dataset.mark = String(m.id);
      return el;
    });
  }
  const after = new Map<Element, Element>();
  for (const m of marks.filter((m) => m.note !== undefined).sort((a, b) => a.start - b.start)) {
    const pieces = root.querySelectorAll(`mark[data-mark="${m.id}"]`);
    const end = pieces[pieces.length - 1];
    if (!end) continue;
    const ref = document.createElement("sup");
    ref.className = "anno note-ref";
    ref.textContent = String(m.n);
    end.after(ref);
    let block: Element = end;
    while (block.parentElement && block.parentElement !== root) block = block.parentElement;
    const card = document.createElement("div");
    card.className = m.sent ? "anno note sent" : "anno note";
    card.dataset.mark = String(m.id);
    card.title = "Click to edit the note";
    const num = document.createElement("span");
    num.className = "note-n";
    num.textContent = String(m.n);
    const text = document.createElement("span");
    text.textContent = m.note || "(empty note)";
    card.append(num, text);
    (after.get(block) ?? block).after(card);
    after.set(block, card);
  }
}

type Model = { value: string; name: string; efforts: string[] };
type SessionRow = { id: string; title: string; at: number; pinned: number }; // pinned: when, 0 = not

type AgentEvent =
  | { type: "ready" }
  | { type: "models"; models: Model[]; defaultModel: string; defaultEffort: string; chosen: boolean }
  | { type: "session"; id: string }
  | { type: "delta"; turn: number; text: string }
  | { type: "tool"; turn: number; name: string }
  | {
      type: "done";
      turn: number;
      isError: boolean;
      error: string | null;
      result: string | null;
      durationMs: number;
    }
  | { type: "history"; sessions: SessionRow[] }
  | { type: "deleted"; id: string }
  | {
      type: "transcript";
      id: string;
      turn: number;
      messages: { role: "user" | "assistant"; text: string; images: number }[];
    }
  | { type: "handoff"; id: string }
  | { type: "compacted"; turn: number; before: number; after: number | null }
  | { type: "context"; turn: number; tokens: number }
  | { type: "tasks"; turn: number; count: number }
  | { type: "wake"; turn: number; compacting?: boolean }
  | { type: "activity"; ids: string[] }
  | { type: "notify" }
  | { type: "error"; turn?: number; message: string };

const send = (cmd: object) => invoke("agent_send", { line: JSON.stringify(cmd) });

// The context gauge shows from CONTEXT_WARN tokens, and turns red from CONTEXT_HIGH.
const CONTEXT_WARN = 100_000;
const CONTEXT_HIGH = 200_000;

const count = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

const tokens = (n: number) => (n < 1000 ? `${n}` : `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`);

function ago(ms: number) {
  const s = (Date.now() - ms) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d ago`;
  return new Date(ms).toLocaleDateString();
}

// Scale a pasted image down to a sensible size and return it base64-encoded.
async function prepareImage(blob: Blob): Promise<Image> {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  let url = canvas.toDataURL("image/png");
  if (url.length * 0.75 > MAX_IMAGE_BYTES) url = canvas.toDataURL("image/jpeg", 0.85);
  const [head, data] = url.split(",");
  return { mediaType: head.slice(5, head.indexOf(";")), data, url };
}

function Markdown(props: { text: string; index?: number; marks?: Paint[] }) {
  let el!: HTMLDivElement;
  // Rebuilt on every change; the marks are painted over the fresh HTML each time.
  createEffect(() => {
    el.innerHTML = DOMPurify.sanitize(marked.parse(props.text, { async: false }) as string);
    if (props.marks?.length) paint(el, props.marks);
  });
  return <div ref={el} class="md" data-msg={props.index} />;
}

function AssistantBody(props: { msg: Msg; index?: number; marks?: Paint[] }) {
  return (
    <>
      <Show when={props.msg.text}>
        <Markdown text={props.msg.text} index={props.index} marks={props.marks} />
      </Show>
      <Show when={props.msg.status}>
        <div class="status">
          <span class="spinner" />
          {props.msg.status}
        </div>
      </Show>
      <Show when={props.msg.error}>
        <div class="error">{props.msg.error}</div>
      </Show>
    </>
  );
}

function UserBody(props: { msg: Msg }) {
  return (
    <div class="bubble">
      <Show when={props.msg.images?.length}>
        <div class="thumbs">
          <For each={props.msg.images}>{(url) => <img src={url} alt="" />}</For>
        </div>
      </Show>
      <Show when={props.msg.imageCount}>
        <div class="image-count">
          {props.msg.imageCount} image{props.msg.imageCount === 1 ? "" : "s"}
        </div>
      </Show>
      {props.msg.text}
      <Show when={props.msg.marked}>
        <div class="marked-chip">{props.msg.marked}</div>
      </Show>
    </div>
  );
}

type Overlay = "none" | "history" | "prefs" | "keys";
type Theme = "system" | "light" | "dark";
type Prefs = { bringForward: boolean; sound: boolean; notification: boolean; theme: Theme };
type Toggle = Exclude<keyof Prefs, "theme">;
const THEMES: { value: Theme; label: string }[] = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

// The settings in ⌘, (stored by the shell), in display order.
const PREF_ROWS: { key: Toggle; title: string; detail: string }[] = [
  {
    key: "bringForward",
    title: "Bring QuickClaude forward",
    detail: "When a chat finishes while you're away, open the panel with that chat.",
  },
  {
    key: "sound",
    title: "Play a sound",
    detail: "A short chime when a chat finishes while you're away.",
  },
  {
    key: "notification",
    title: "Show a notification",
    detail: "A macOS notification with the reply. Skipped when QuickClaude comes forward.",
  },
];

// The shortcuts screen (⌘/), in the Preferences layout. The panel shows no key hints of its own.
const KEY_SECTIONS: { title: string; rows: { keys: string[]; title: string }[] }[] = [
  {
    title: "Anywhere",
    rows: [
      { keys: ["⌥Space"], title: "Show or hide QuickClaude" },
      { keys: ["⌘/"], title: "Keyboard shortcuts" },
      { keys: ["⌘,"], title: "Preferences" },
      { keys: ["⌘F"], title: "Full screen" },
      { keys: ["⌘T"], title: "Keep on top" },
      { keys: ["⌘N"], title: "New chat" },
      { keys: ["⌘⇧H"], title: "Continue in a terminal" },
      { keys: ["⌘."], title: "Stop the answer" },
      { keys: ["⇥"], title: "Next model" },
      { keys: ["⇧⇥"], title: "Next effort" },
      { keys: ["esc"], title: "Go back a step, then hide" },
      { keys: ["⇧esc"], title: "Hide, keeping everything as it is" },
    ],
  },
  {
    title: "Asking",
    rows: [
      { keys: ["↩"], title: "Ask a new question" },
      { keys: ["⌘↩"], title: "Follow up as a chat" },
      { keys: ["⇧↩"], title: "New line" },
      { keys: ["↓"], title: "Open history (on an empty input)" },
      { keys: ["⌘V"], title: "Paste an image" },
      { keys: ["⌫"], title: "Remove the last image (on an empty input)" },
      { keys: ["drop"], title: "Drop files to insert their paths" },
    ],
  },
  {
    title: "Highlights (select text in a reply)",
    rows: [
      { keys: ["h"], title: "Highlight it (again to remove)" },
      { keys: ["n"], title: "Highlight it with a note; click a note to edit it" },
      { keys: ["⇧C"], title: "Clear all highlights and notes" },
      { keys: ["⌘↩"], title: "Send them with your follow-up (sent ones fade)" },
    ],
  },
  {
    title: "History",
    rows: [
      { keys: ["↑", "↓"], title: "Choose a chat" },
      { keys: ["↩"], title: "Open it" },
      { keys: ["⇧↑", "⇧↓"], title: "Select several (or ⇧click, ⌘click)" },
      { keys: ["⌘A"], title: "Select all" },
      { keys: ["⌘P"], title: "Pin or unpin (pinned chats stay on top)" },
      { keys: ["⌫"], title: "Delete (press twice)" },
      { keys: ["esc"], title: "Clear the selection, then close" },
    ],
  },
  {
    title: "Preferences",
    rows: [
      { keys: ["↑", "↓"], title: "Choose a setting" },
      { keys: ["space"], title: "Turn it on or off" },
      { keys: ["←", "→"], title: "Change the theme" },
    ],
  },
];

export default function App() {
  const [messages, setMessages] = createStore<Msg[]>([]);
  const [mode, setMode] = createSignal<"compact" | "chat">("compact");
  const [overlay, setOverlay] = createSignal<Overlay>("none");
  const [busy, setBusy] = createSignal(false);
  const [draft, setDraft] = createSignal("");
  const [notice, setNotice] = createSignal("");
  const [attachments, setAttachments] = createSignal<Image[]>([]);
  const [sessionId, setSessionId] = createSignal<string | null>(null);
  // The open chat's name from history; a chat started here is named by its first question.
  const [chatTitle, setChatTitle] = createSignal("");
  const chatName = () => chatTitle() || messages.find((m) => m.role === "user")?.text || "Chat";
  const [models, setModels] = createSignal<Model[]>([]);
  const [model, setModel] = createSignal("");
  const [effort, setEffort] = createSignal("");
  // Which half of the label just changed; drives the highlight, dots and placeholder.
  const [flash, setFlash] = createSignal<"model" | "effort" | null>(null);
  const [sessions, setSessions] = createSignal<SessionRow[] | null>(null);
  const [selected, setSelected] = createSignal(0);
  // Background tasks of the open chat, and every chat with something running (history lamps).
  const [tasks, setTasks] = createSignal(0);
  const [active, setActive] = createSignal<Set<string>>(new Set());
  // The open chat's context size; unknown (null) until its first turn here.
  const [context, setContext] = createSignal<number | null>(null);
  // Busy, or waiting on background work that will reply by itself.
  const working = () => busy() || tasks() > 0;
  let turn = 0;
  let stashedDraft = "";
  let input!: HTMLTextAreaElement;
  let panel!: HTMLDivElement;
  let logo!: SVGSVGElement;
  let list: HTMLDivElement | undefined;
  let historyList: HTMLDivElement | undefined;
  let keysList: HTMLDivElement | undefined;

  const win = getCurrentWindow();
  const last = () => messages.length - 1;
  const lastAnswer = () => messages[last()];
  const tall = () => overlay() === "history" || overlay() === "keys" || (mode() === "chat" && overlay() === "none");

  // --- model & effort ---
  const modelName = () => models().find((m) => m.value === model())?.name ?? "";

  // The sidecar resolves the defaults: your last pick, else your Claude Code settings.
  function onModels(list: Model[], defaultModel: string, defaultEffort: string) {
    setModels(list);
    const m = list.find((x) => x.value === (model() || defaultModel)) ?? list[0];
    if (!m) return;
    setModel(m.value);
    const e = effort() || defaultEffort;
    setEffort(m.efforts.includes(e) ? e : (m.efforts[m.efforts.length - 1] ?? ""));
  }

  // Tab cycles the model, ⇧Tab the effort. The choice is saved by the sidecar.
  let flashTimer = 0;
  function cycle(kind: "model" | "effort") {
    const list = models();
    const i = list.findIndex((m) => m.value === model());
    if (i < 0) return;
    if (kind === "model") {
      const next = list[(i + 1) % list.length];
      setModel(next.value);
      if (!next.efforts.includes(effort())) setEffort(next.efforts[next.efforts.length - 1] ?? "");
    } else {
      const efforts = list[i].efforts;
      if (efforts.length < 2) return;
      setEffort(efforts[(efforts.indexOf(effort()) + 1) % efforts.length]);
    }
    send({ type: "choice", model: model(), effort: effort() });
    setFlash(kind);
    clearTimeout(flashTimer);
    flashTimer = window.setTimeout(() => setFlash(null), 1200);
  }

  // --- history ---
  // Pinned chats first (newest pin on top), then the rest by last activity.
  const ordered = createMemo(() =>
    [...(sessions() ?? [])].sort((a, b) => b.pinned - a.pinned || (a.pinned ? 0 : b.at - a.at)),
  );
  const filtered = createMemo(() => {
    const q = draft().trim().toLowerCase();
    const all = ordered();
    return q ? all.filter((s) => s.title.toLowerCase().includes(q)) : all;
  });
  // Chats picked with ⇧↑↓, ⇧click, ⌘click or ⌘A; empty means just the row under the cursor.
  const [picked, setPicked] = createSignal<Set<string>>(new Set());
  let anchor = 0; // where a ⇧ range starts
  const targets = () => {
    const p = picked();
    return p.size ? filtered().filter((r) => p.has(r.id)) : filtered().slice(selected(), selected() + 1);
  };
  const isTarget = (id: string) => (picked().size ? picked().has(id) : filtered()[selected()]?.id === id);
  // Deleting takes two presses: the first marks the rows, the second deletes them from Claude Code.
  const [armed, setArmed] = createSignal(false);
  // A new search starts at the top; a deletion keeps your place in the list.
  createEffect(
    on(draft, () => {
      if (overlay() !== "history") return;
      setSelected(0);
      setPicked(new Set<string>());
    }),
  );
  createEffect(() => {
    const n = filtered().length;
    if (selected() >= n) setSelected(Math.max(n - 1, 0));
  });
  createEffect(on([selected, picked, draft, overlay], () => setArmed(false)));

  function pickRange(to: number) {
    const [a, b] = [Math.min(anchor, to), Math.max(anchor, to)];
    setPicked(new Set(filtered().slice(a, b + 1).map((r) => r.id)));
  }

  function moveCursor(step: number, extend: boolean) {
    const n = filtered().length;
    if (!n) return;
    if (extend && !picked().size) anchor = selected();
    const to = extend ? Math.min(Math.max(selected() + step, 0), n - 1) : (selected() + step + n) % n;
    setSelected(to);
    if (extend) pickRange(to);
    else setPicked(new Set<string>());
  }

  function onRowClick(e: MouseEvent, i: number, row: SessionRow) {
    if (e.shiftKey) {
      if (!picked().size) anchor = selected();
      setSelected(i);
      pickRange(i);
    } else if (e.metaKey) {
      const p = new Set(picked().size ? picked() : [filtered()[selected()]?.id].filter(Boolean) as string[]);
      if (p.has(row.id)) p.delete(row.id);
      else p.add(row.id);
      setPicked(p);
      setSelected(i);
      anchor = i;
    } else openSession(row);
  }
  createEffect(() => {
    const i = selected();
    if (overlay() === "history") historyList?.children[i]?.scrollIntoView({ block: "nearest" });
  });

  // Rows that change place slide from where they were to where they land (FLIP).
  function flip(change: () => void, lift: Set<string>) {
    const rows = () => [...(historyList?.querySelectorAll<HTMLElement>("[data-id]") ?? [])];
    const before = new Map(rows().map((el) => [el.dataset.id!, el.getBoundingClientRect().top]));
    change();
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    for (const el of rows()) {
      const from = before.get(el.dataset.id!);
      const dy = from === undefined ? 0 : from - el.getBoundingClientRect().top;
      if (Math.abs(dy) < 1) continue;
      // The pinned rows pass over the others on their way up.
      if (lift.has(el.dataset.id!)) el.classList.add("lifted");
      el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], {
        duration: 420,
        easing: "cubic-bezier(0.3, 0.7, 0.3, 1)",
      }).finished.then(
        () => el.classList.remove("lifted"),
        () => el.classList.remove("lifted"),
      );
    }
  }

  // ⌘P pins the chat under the cursor (or the selected ones); if all are pinned, it unpins them.
  // Their "when" column says "Pinned" or "Unpinned" for a moment.
  const [pinFlash, setPinFlash] = createSignal<{ ids: Set<string>; pinned: boolean } | null>(null);
  let pinTimer = 0;
  function togglePin() {
    const rows = targets();
    if (!rows.length) return;
    const pinned = rows.some((r) => !r.pinned);
    const ids = rows.map((r) => r.id);
    const now = Date.now();
    flip(() => {
      setSessions((all) =>
        all?.map((r) => (ids.includes(r.id) ? { ...r, pinned: pinned ? now - ids.indexOf(r.id) : 0 } : r)) ?? null,
      );
      // The cursor follows the chat it was on.
      setSelected(Math.max(filtered().findIndex((r) => r.id === ids[0]), 0));
    }, new Set(pinned ? ids : []));
    setPinFlash({ ids: new Set(ids), pinned });
    clearTimeout(pinTimer);
    pinTimer = window.setTimeout(() => setPinFlash(null), 1400);
    send({ type: "pin", ids, pinned });
  }

  function deleteSelected() {
    const rows = targets();
    if (!rows.length) return;
    if (!armed()) return setArmed(true);
    setArmed(false);
    setPicked(new Set<string>());
    // Land on the chat that follows the first deleted one.
    setSelected(filtered().indexOf(rows[0]));
    for (const row of rows) send({ type: "delete", id: row.id });
  }

  // --- preferences (⌘,) ---
  const [prefs, setPrefs] = createSignal<Prefs | null>(null);
  const [prefRow, setPrefRow] = createSignal(0);
  async function openPrefs() {
    setPrefRow(0);
    setOverlay("prefs");
    setPrefs(await invoke<Prefs>("get_prefs"));
  }
  function savePrefs(next: Prefs) {
    setPrefs(next);
    invoke("set_prefs", { new: next }).catch((e) => setNotice(`Couldn't save preferences: ${e}`));
  }
  function togglePref(key: Toggle) {
    const p = prefs();
    if (!p) return;
    const next = { ...p, [key]: !p[key] };
    savePrefs(next);
    if (key === "sound" && next.sound) invoke("preview_sound");
  }
  // The theme row sits after the switches; step moves through System, Light, Dark.
  const themeRow = PREF_ROWS.length;
  function stepTheme(step: number) {
    const p = prefs();
    if (!p) return;
    const i = THEMES.findIndex((t) => t.value === p.theme);
    savePrefs({ ...p, theme: THEMES[(Math.max(i, 0) + step + THEMES.length) % THEMES.length].value });
  }
  function onPrefsKey(e: KeyboardEvent) {
    const n = PREF_ROWS.length + 1;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setPrefRow((i) => (i + (e.key === "ArrowDown" ? 1 : n - 1)) % n);
    } else if (prefRow() === themeRow && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      stepTheme(e.key === "ArrowRight" ? 1 : -1);
    } else if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (prefRow() === themeRow) stepTheme(1);
      else togglePref(PREF_ROWS[prefRow()].key);
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOverlay("none");
    }
  }

  function onKeysKey(e: KeyboardEvent) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      keysList?.scrollBy({ top: e.key === "ArrowDown" ? 80 : -80, behavior: "smooth" });
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOverlay("none");
    }
  }

  function showHistory() {
    stashedDraft = draft();
    setDraft("");
    setSessions(null);
    setSelected(0);
    setOverlay("history");
    send({ type: "history" });
  }

  function closeOverlay() {
    if (overlay() === "history") {
      setDraft(stashedDraft);
      queueMicrotask(autosize);
    }
    setOverlay("none");
  }

  function openSession(row: SessionRow | undefined) {
    if (!row) return;
    turn++;
    dropMarks();
    setOverlay("none");
    setDraft("");
    setAttachments([]);
    setMessages([{ role: "assistant", text: "", status: "Opening chat…" }]);
    setBusy(false);
    setNotice("");
    setSessionId(row.id);
    setChatTitle(row.title);
    setTasks(0);
    setContext(null);
    setMode("chat");
    send({ type: "open", id: row.id, turn });
  }

  // --- streaming: batch deltas into one store update per frame ---
  let pending = "";
  let frame = 0;
  const flush = () => {
    cancelAnimationFrame(frame);
    frame = 0;
    if (!pending || last() < 0) return;
    const chunk = pending;
    pending = "";
    setMessages(last(), (m) => ({ text: m.text + chunk, status: undefined }));
  };

  function onAgent(ev: AgentEvent) {
    if ("turn" in ev && ev.turn !== undefined && ev.turn !== turn) return;
    switch (ev.type) {
      case "models":
        onModels(ev.models, ev.defaultModel, ev.defaultEffort);
        break;
      case "session":
        setSessionId(ev.id);
        break;
      case "delta":
        pending += ev.text;
        if (!frame) frame = requestAnimationFrame(flush);
        break;
      case "tool":
        flush();
        setMessages(last(), "status", `Using ${ev.name.replace(/^mcp__[^_]+__/, "")}…`);
        break;
      case "done": {
        flush();
        const m = lastAnswer();
        setMessages(last(), {
          status: undefined,
          text: m.text || ev.result || "",
          error: ev.isError ? ev.error || "Something went wrong." : undefined,
          meta: `${(ev.durationMs / 1000).toFixed(1)}s`,
        });
        setBusy(false);
        break;
      }
      case "history":
        setSessions(ev.sessions);
        break;
      case "tasks":
        setTasks(ev.count);
        break;
      case "activity":
        setActive(new Set(ev.ids));
        break;
      case "wake": {
        // Claude carries on by itself (a background task finished), or a reopened chat is
        // mid-turn: continue its last answer. A /compact gets its own line, as when the chip started it.
        flush();
        const status = ev.compacting ? "Compacting…" : "Working…";
        if (lastAnswer()?.role === "assistant" && !ev.compacting) {
          if (lastAnswer().text) pending = "\n\n";
          setMessages(last(), { status, meta: undefined, error: undefined });
        } else setMessages(messages.length, { role: "assistant", text: "", status });
        setBusy(true);
        break;
      }
      case "deleted":
        setSessions((all) => all?.filter((s) => s.id !== ev.id) ?? null);
        // The open chat was the one deleted: it no longer exists to follow up on.
        if (sessionId() === ev.id) {
          turn++;
          dropMarks();
          setTasks(0);
          setMessages([]);
          setBusy(false);
          setMode("compact");
          setSessionId(null);
          setContext(null);
        }
        break;
      case "transcript":
        setMessages(
          ev.messages.map((m) => ({ role: m.role, text: m.text, imageCount: m.images || undefined })),
        );
        break;
      case "compacted":
        setMessages(
          last(),
          "text",
          ev.after === null
            ? `_Compacted from ${tokens(ev.before)} tokens._`
            : `_Compacted: ${tokens(ev.before)} → ${tokens(ev.after)} tokens._`,
        );
        break;
      case "context":
        setContext(ev.tokens);
        break;
      case "handoff":
        clear();
        win.hide();
        break;
      case "error":
        if (busy() && ev.turn !== undefined) {
          flush();
          setMessages(last(), { status: undefined, error: ev.message });
        } else {
          setNotice(ev.message);
        }
        setBusy(false);
        break;
    }
  }

  function ask(followUp: boolean) {
    const text = draft().trim();
    const images = attachments();
    // New highlights and notes go only with a follow-up; a new question drops them.
    const unsent = followUp ? marks().filter((m) => !m.sent) : [];
    if ((!text && !images.length && !unsent.length) || (busy() && followUp)) return;
    turn++;
    pending = "";
    if (!followUp) {
      setMessages([]);
      setSessionId(null);
      setChatTitle("");
      setTasks(0);
      setContext(null);
      dropMarks();
    }
    const notes = unsent.filter((m) => m.note).length;
    setMessages(messages.length, {
      role: "user",
      text,
      images: images.map((i) => i.url),
      marked: unsent.length
        ? `${count(unsent.length, "highlight")}${notes ? ` · ${count(notes, "note")}` : ""}`
        : undefined,
    });
    setMarks((ms) => ms.map((m) => (unsent.includes(m) ? { ...m, sent: true } : m)));
    setMessages(messages.length, { role: "assistant", text: "", status: "Thinking…" });
    setBusy(true);
    setNotice("");
    setDraft("");
    setAttachments([]);
    autosize();
    send({
      type: "ask",
      text,
      images: images.map(({ mediaType, data }) => ({ mediaType, data })),
      annotations: [...unsent]
        .sort((a, b) => a.msg - b.msg || a.start - b.start)
        .map((m) => ({ quote: m.quote, note: m.note })),
      followUp,
      turn,
      model: model(),
      effort: effort(),
    });
  }

  function clear() {
    turn++;
    dropMarks();
    setTasks(0);
    setContext(null);
    setMessages([]);
    setBusy(false);
    setMode("compact");
    setOverlay("none");
    setDraft("");
    setAttachments([]);
    setSessionId(null);
    setChatTitle("");
    setNotice("");
  }

  function reset() {
    send({ type: "reset" });
    clear();
    input.focus();
  }

  // ⌘N: an empty input, and the panel glides back to the centre of its screen.
  function newChat() {
    reset();
    if (!full()) invoke("center_panel");
  }

  function toTerminal() {
    if (!sessionId() || working()) return;
    send({ type: "handoff" });
  }

  // /compact in the same session; the chat stays on screen with a note under it.
  function compact() {
    if (!sessionId() || working()) return;
    turn++;
    pending = "";
    setMode("chat");
    setMessages(messages.length, { role: "assistant", text: "", status: "Compacting…" });
    setBusy(true);
    setNotice("");
    send({ type: "compact", turn });
  }

  // The context chip compacts on a second click; the first asks "Compact 398k?".
  // Esc, typing, another chat or 5 s without the second click cancel it.
  const [compactArmed, setCompactArmed] = createSignal(false);
  let compactTimer = 0;
  function disarmCompact() {
    clearTimeout(compactTimer);
    setCompactArmed(false);
  }
  function onChipClick() {
    if (compactArmed()) {
      disarmCompact();
      compact();
    } else {
      setCompactArmed(true);
      compactTimer = window.setTimeout(() => setCompactArmed(false), 5000);
    }
  }
  createEffect(on([draft, sessionId], disarmCompact, { defer: true }));

  // --- highlights & notes: select text in a reply, then h, n or ⇧C ---
  const [marks, setMarks] = createSignal<Mark[]>([]);
  // The note being written in the input: its mark, whether n just made it, and the draft put aside.
  const [noting, setNoting] = createSignal<{ id: number; fresh: boolean } | null>(null);
  let noteStash = "";
  let markSeq = 0;
  const [markFlash, setMarkFlash] = createSignal("");
  let markFlashTimer = 0;
  const notingQuote = () => {
    const q = marks().find((m) => m.id === noting()?.id)?.quote ?? "";
    return q.length > 40 ? `${q.slice(0, 40).trimEnd()}…` : q;
  };

  // One reply's marks, each note numbered by its place in the chat.
  function marksFor(i: number): Paint[] {
    const all = marks();
    const noted = all.filter((m) => m.note !== undefined).sort((a, b) => a.msg - b.msg || a.start - b.start);
    return all.filter((m) => m.msg === i).map((m) => ({ ...m, n: noted.indexOf(m) + 1 || undefined }));
  }

  // The selection, if it lies within one reply: which reply, its offsets and the quoted text.
  function selectionSpot() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const r = sel.getRangeAt(0);
    const host = (n: Node) => (n instanceof Element ? n : n.parentElement)?.closest<HTMLElement>(".md");
    const root = host(r.startContainer);
    if (!root || root !== host(r.endContainer) || root.dataset.msg === undefined) return null;
    const clone = root.cloneNode(true) as HTMLElement;
    clone.querySelectorAll(ANNO).forEach((el) => el.remove());
    const text = clone.textContent ?? "";
    let start = offsetIn(root, r.startContainer, r.startOffset);
    let end = offsetIn(root, r.endContainer, r.endOffset);
    while (start < end && /\s/.test(text[start])) start++;
    while (end > start && /\s/.test(text[end - 1])) end--;
    if (start >= end) return null;
    return { msg: Number(root.dataset.msg), start, end, quote: text.slice(start, end) };
  }

  // h highlights the selection, or removes the highlights it touches; n adds a note
  // (or edits the note of the highlight it touches).
  function markSelection(withNote: boolean) {
    const spot = selectionSpot();
    if (!spot) return false;
    window.getSelection()?.removeAllRanges();
    const hit = marks().filter((m) => m.msg === spot.msg && m.start < spot.end && spot.start < m.end);
    if (hit.length && !withNote) {
      setMarks((ms) => ms.filter((m) => !hit.includes(m)));
      input.focus();
    } else if (hit.length) {
      startNote(hit[0].id, false);
    } else {
      const m: Mark = { id: ++markSeq, ...spot, sent: false, note: withNote ? "" : undefined };
      setMarks((ms) => [...ms, m]);
      if (withNote) startNote(m.id, true);
      else input.focus();
    }
    return true;
  }

  // The note is written in the input; what was typed there comes back after.
  function startNote(id: number, fresh: boolean) {
    if (noting()) finishNote(true);
    noteStash = draft();
    setNoting({ id, fresh });
    setDraft(marks().find((m) => m.id === id)?.note ?? "");
    input.focus();
    queueMicrotask(autosize);
  }

  // ↩ saves (an empty note leaves a plain highlight); Esc drops a new one, or leaves an old one as it was.
  function finishNote(save: boolean) {
    const n = noting();
    if (!n) return;
    const text = draft().trim() || undefined;
    if (save) setMarks((ms) => ms.map((m) => (m.id === n.id && m.note !== text ? { ...m, note: text, sent: false } : m)));
    else if (n.fresh) setMarks((ms) => ms.filter((m) => m.id !== n.id));
    setNoting(null);
    setDraft(noteStash);
    noteStash = "";
    queueMicrotask(autosize);
  }

  // ⇧C: clear them all; the placeholder says how many went.
  function clearMarks() {
    if (noting()) finishNote(false);
    const all = marks();
    if (!all.length) return;
    const notes = all.filter((m) => m.note).length;
    setMarks([]);
    setMarkFlash(`Cleared ${count(all.length, "highlight")}${notes ? `, ${count(notes, "note")}` : ""}`);
    clearTimeout(markFlashTimer);
    markFlashTimer = window.setTimeout(() => setMarkFlash(""), 1600);
  }

  // Leaving the chat drops them; they belong to it.
  function dropMarks() {
    setMarks([]);
    setNoting(null);
    noteStash = "";
  }

  async function onPaste(e: ClipboardEvent) {
    const items = [...(e.clipboardData?.items ?? [])];
    const files = items.filter((i) => i.kind === "file" && i.type.startsWith("image/"));
    if (!files.length) return;
    if (!items.some((i) => i.type === "text/plain")) e.preventDefault();
    for (const item of files) {
      const blob = item.getAsFile();
      if (!blob) continue;
      try {
        const img = await prepareImage(blob);
        setAttachments((a) => [...a, img]);
      } catch {
        setNotice("Couldn't read that image.");
      }
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (overlay() !== "none") {
      const del = e.key === "Delete" || (e.key === "Backspace" && (e.metaKey || !draft()));
      if (e.key === "Tab") e.preventDefault();
      else if (del) {
        e.preventDefault();
        deleteSelected();
      }
      else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        moveCursor(e.key === "ArrowDown" ? 1 : -1, e.shiftKey);
      } else if (e.metaKey && e.key === "a") {
        e.preventDefault();
        setPicked(new Set(filtered().map((r) => r.id)));
      } else if (e.metaKey && e.key === "p") {
        e.preventDefault();
        togglePin();
      } else if (e.key === "Enter") {
        e.preventDefault();
        openSession(filtered()[selected()]);
      } else if (e.key === "Escape") {
        e.preventDefault();
        // First drop a multi-selection, then leave the list.
        if (picked().size) setPicked(new Set<string>());
        else closeOverlay();
      }
      return;
    }
    if (noting()) {
      // The input holds a note: ↩ saves it, Esc cancels; the panel's ⌘ keys still work.
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        return finishNote(true);
      }
      if (e.key === "Escape") {
        e.preventDefault();
        return finishNote(false);
      }
      if (!e.metaKey) {
        if (e.key === "Tab" || e.key === "ArrowDown") e.preventDefault();
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (mode() === "chat") ask(true);
      else if (e.metaKey && messages.length) {
        setMode("chat");
        ask(true);
      } else ask(false);
    } else if (e.key === "Tab") {
      e.preventDefault();
      cycle(e.shiftKey ? "effort" : "model");
    } else if (e.key === "ArrowDown" && !draft()) {
      e.preventDefault();
      showHistory();
    } else if (e.key === "Escape") {
      // One step back per press: cancel an asked compact, clear what's typed (and pasted),
      // close the chat, then hide.
      e.preventDefault();
      if (compactArmed()) disarmCompact();
      else if (draft() || attachments().length) {
        setDraft("");
        setAttachments([]);
        autosize();
      } else if (mode() === "chat") {
        // Back to an empty input. A chat that's still working keeps running in the background
        // (green lamp in history, notification when it replies).
        reset();
      } else win.hide();
    } else if (e.key === "Backspace" && !draft() && attachments().length) {
      e.preventDefault();
      setAttachments((a) => a.slice(0, -1));
    } else if (e.metaKey && e.key === "t") {
      e.preventDefault();
      toggleOnTop();
    } else if (e.metaKey && e.key === ".") {
      e.preventDefault();
      if (busy()) send({ type: "stop" });
    }
  }

  // Keep on top: the panel stays up when another app takes focus; Esc still hides it.
  // A faint accent ring marks it; toggling turns the spark a quarter and names the new state.
  const [onTop, setOnTop] = createSignal(false);
  const [onTopFlash, setOnTopFlash] = createSignal(false);
  let onTopTimer = 0;
  function toggleOnTop() {
    const next = !onTop();
    setOnTop(next);
    invoke("set_on_top", { on: next });
    logo.animate([{ transform: "rotate(0deg) scale(1)" }, { transform: "rotate(45deg) scale(1.18)" }, { transform: "rotate(90deg) scale(1)" }], {
      duration: 420,
      easing: "cubic-bezier(0.3, 0.7, 0.3, 1)",
    });
    setOnTopFlash(true);
    clearTimeout(onTopTimer);
    onTopTimer = window.setTimeout(() => setOnTopFlash(false), 1400);
  }

  // Full screen (⌘F, in any view): the panel fills the screen with a centred reading column.
  // The shell places it and puts it back; until then the panel stops sizing itself.
  const [full, setFull] = createSignal(false);
  function toggleFull() {
    const next = !full();
    setFull(next);
    if (next) invoke("set_full", { full: true, width: WIDTH, height: lastHeight });
    else {
      // Measured with the full-screen layout gone, so the panel returns at its normal size.
      lastHeight = wantedHeight();
      invoke("set_full", { full: false, width: WIDTH, height: lastHeight });
    }
  }

  // Nothing but an empty input: the shell reopens a fresh panel centred, wherever it was left.
  const fresh = createMemo(
    () => mode() === "compact" && messages.length === 0 && overlay() === "none" && !draft() && attachments().length === 0,
  );
  createEffect(() => invoke("set_fresh", { fresh: fresh() }));

  // Files dropped on the panel go into the input as their full paths, at the cursor.
  function insertPaths(paths: string[]) {
    if (!paths.length) return;
    const text = paths.join(" ");
    const start = input.selectionStart ?? draft().length;
    const end = input.selectionEnd ?? start;
    const before = draft().slice(0, start);
    const after = draft().slice(end);
    const pad = before && !/\s$/.test(before) ? " " : "";
    const tail = after && !/^\s/.test(after) ? " " : "";
    setDraft(before + pad + text + tail + after);
    input.value = draft();
    const caret = (before + pad + text + tail).length;
    input.focus();
    input.setSelectionRange(caret, caret);
    autosize();
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
  }

  // --- window sizing: compact hugs its content; chat and history are a fixed tall panel ---
  let lastHeight = 0;
  const wantedHeight = () => (tall() ? CHAT_HEIGHT : Math.min(Math.ceil(panel.offsetHeight), COMPACT_MAX));
  const fit = () => {
    if (full()) return;
    const h = wantedHeight();
    if (h !== lastHeight) {
      lastHeight = h;
      win.setSize(new LogicalSize(WIDTH, h));
    }
  };
  // The input bar moves between layouts when the view changes, so refocus it.
  createEffect(() => {
    mode();
    overlay();
    queueMicrotask(() => {
      fit();
      input.focus();
    });
  });

  // Keep the chat scrolled to the newest text while it streams.
  createEffect(() => {
    const m = messages[messages.length - 1];
    m?.text;
    m?.status;
    messages.length;
    overlay();
    if (mode() === "chat" && list) list.scrollTop = list.scrollHeight;
  });

  // ⇧Esc: hide and change nothing, so the panel reopens exactly as it was. Caught while
  // capturing, before the input's and history's own Esc steps can run.
  function onShiftEsc(e: KeyboardEvent) {
    if (!e.shiftKey || e.key !== "Escape") return;
    e.preventDefault();
    e.stopPropagation();
    win.hide();
  }

  // Selecting text in the chat moves focus off the input; the panel's keys should still work.
  function onWindowKeyDown(e: KeyboardEvent) {
    // ⌘N works from every screen, history, preferences and shortcuts included.
    if (e.metaKey && e.key === "n") {
      e.preventDefault();
      newChat();
      return;
    }
    if (e.metaKey && e.code === "KeyF") {
      e.preventDefault();
      toggleFull();
      return;
    }
    if (e.metaKey && e.key === ",") {
      e.preventDefault();
      if (overlay() === "prefs") setOverlay("none");
      else {
        if (overlay() === "history") closeOverlay();
        openPrefs();
      }
      return;
    }
    if (e.metaKey && e.key === "/") {
      e.preventDefault();
      if (overlay() === "keys") setOverlay("none");
      else {
        if (overlay() === "history") closeOverlay();
        setOverlay("keys");
      }
      return;
    }
    if (overlay() === "prefs") return onPrefsKey(e);
    if (overlay() === "keys") return onKeysKey(e);
    // With text selected in a reply (focus is then off the input): h, n, and ⇧C for all.
    if (overlay() === "none" && e.target !== input && !noting() && !e.metaKey && !e.ctrlKey && !e.altKey) {
      if ((e.key === "h" || e.key === "n") && markSelection(e.key === "n")) return e.preventDefault();
      if (e.key === "C" && marks().length) {
        e.preventDefault();
        return clearMarks();
      }
    }
    if (e.target === input) return;
    const panelKey = e.key === "Escape" || e.key === "Tab" || (e.metaKey && ["t", ".", "p"].includes(e.key));
    if (!panelKey) return;
    window.getSelection()?.removeAllRanges();
    input.focus();
    onKeyDown(e);
  }

  onMount(() => {
    window.addEventListener("keydown", onShiftEsc, true);
    window.addEventListener("keydown", onWindowKeyDown);
    onCleanup(() => {
      window.removeEventListener("keydown", onShiftEsc, true);
      window.removeEventListener("keydown", onWindowKeyDown);
    });
    const observer = new ResizeObserver(fit);
    observer.observe(panel);
    const unlisten = Promise.all([
      listen<string>("agent", (e) => onAgent(JSON.parse(e.payload))),
      listen("open-prefs", () => openPrefs()),
      // ⌘⇧H arrives from the app menu (the shell), not as a key event.
      listen("to-terminal", () => overlay() === "none" && toTerminal()),
      getCurrentWebview().onDragDropEvent((e) => {
        if (e.payload.type === "drop") insertPaths(e.payload.paths);
      }),
      listen<string>("open-chat", (e) => {
        if (sessionId() !== e.payload) openSession({ id: e.payload, title: "", at: 0, pinned: 0 });
        else setMode("chat");
      }),
      listen("panel-shown", () => {
        // The panel comes back exactly as it was left: answer, chat, open screen, draft and cursor.
        if (overlay() === "history") {
          // A delete armed before hiding shouldn't fire on the first key after; the list may be stale.
          setArmed(false);
          send({ type: "history" });
        }
        input.focus();
      }),
    ]);
    onCleanup(() => {
      observer.disconnect();
      unlisten.then((fns) => fns.forEach((u) => u()));
    });
    send({ type: "hello" });
    input.focus();
  });

  // Drag the panel by any spot that isn't a control or selectable text; it reopens where it's left.
  function onMouseDown(e: MouseEvent) {
    if (e.button !== 0 || full()) return;
    if ((e.target as HTMLElement).closest("button, a, textarea, input, .pref, .answer, .list")) return;
    e.preventDefault();
    win.startDragging();
  }

  // Open links in the default browser instead of inside the panel.
  function onClick(e: MouseEvent) {
    // A note card under a reply: edit its note in the input.
    const card = (e.target as HTMLElement).closest<HTMLElement>(".note[data-mark]");
    if (card) return startNote(Number(card.dataset.mark), false);
    const a = (e.target as HTMLElement).closest("a");
    if (a?.href) {
      e.preventDefault();
      openUrl(a.href);
    }
  }

  const bar = (
    <div class="bar-wrap">
      <Show when={attachments().length && overlay() !== "history"}>
        <div class="attachments">
          <For each={attachments()}>
            {(img, i) => (
              <button
                class="attachment"
                title="Remove"
                onClick={() => setAttachments((a) => a.filter((_, j) => j !== i()))}
              >
                <img src={img.url} alt="" />
                <span>×</span>
              </button>
            )}
          </For>
        </div>
      </Show>
      <div class="bar">
        <svg ref={logo} class="logo" viewBox="2 0 20 20" aria-hidden="true">
          <path d="M12 2l1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8z" />
        </svg>
        <textarea
          ref={input}
          rows={1}
          value={draft()}
          placeholder={
            overlay() === "history"
              ? "Search chats…"
              : noting()
              ? `Note on “${notingQuote()}”`
              : markFlash()
              ? markFlash()
              : onTopFlash()
              ? onTop()
                ? "Keep on top: on"
                : "Keep on top: off"
              : flash()
                ? `Ask ${modelName()} · ${effort()}…`
                : mode() === "chat"
                  ? "Ask a follow-up…"
                  : "Ask Claude…"
          }
          onInput={(e) => {
            setDraft(e.currentTarget.value);
            autosize();
          }}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          spellcheck={false}
        />
        <Show when={busy()}>
          <button class="stop" title="Stop (⌘.)" onClick={() => send({ type: "stop" })}>
            ■
          </button>
        </Show>
        <Show when={modelName() && overlay() !== "history"}>
          <div class="model-label">
            <button class="part model" title="Next model (⇥)" onClick={() => cycle("model")}>
              <Show when={model()} keyed>
                {(_) => <span classList={{ flash: flash() === "model" }}>{modelName()}</span>}
              </Show>
              <Show when={flash() === "model"}>
                <span class="dots">
                  <For each={models()}>{(m) => <i classList={{ on: m.value === model() }} />}</For>
                </span>
              </Show>
            </button>
            <span class="sep">·</span>
            <button class="part" title="Next effort (⇧⇥)" onClick={() => cycle("effort")}>
              <Show when={effort()} keyed>
                {(e) => <span classList={{ flash: flash() === "effort" }}>{e}</span>}
              </Show>
            </button>
          </div>
        </Show>
      </div>
    </div>
  );

  // Claude ended its turn but left work running; it will reply by itself when that finishes.
  const waiting = () => (
    <Show when={!busy() && tasks() > 0}>
      <div class="status">
        <span class="lamp" />
        <span>
          {tasks() === 1
            ? "Waiting on a background task; Claude will reply when it finishes."
            : `Waiting on ${tasks()} background tasks; Claude will reply when they finish.`}
        </span>
      </div>
    </Show>
  );

  const footer = (hints: string[]): JSX.Element => (
    <div class="footer">
      <For each={hints}>{(h) => <span class="hint">{h}</span>}</For>
    </div>
  );

  const prefsView = (
    <>
      <div class="prefs-head">Preferences</div>
      <div class="prefs">
        <For each={PREF_ROWS}>
          {(row, i) => (
            <button
              class="pref"
              classList={{ selected: i() === prefRow() }}
              onMouseMove={() => setPrefRow(i())}
              onClick={() => togglePref(row.key)}
            >
              <span class="pref-text">
                <span class="pref-title">{row.title}</span>
                <span class="pref-detail">{row.detail}</span>
              </span>
              <span class="switch" classList={{ on: !!prefs()?.[row.key] }} role="switch" aria-checked={!!prefs()?.[row.key]}>
                <span class="knob" />
              </span>
            </button>
          )}
        </For>
        <div
          class="pref"
          classList={{ selected: prefRow() === themeRow }}
          onMouseMove={() => setPrefRow(themeRow)}
        >
          <span class="pref-text">
            <span class="pref-title">Theme</span>
            <span class="pref-detail">System follows your Mac's light or dark appearance.</span>
          </span>
          <span class="segments" role="radiogroup">
            <For each={THEMES}>
              {(t) => (
                <span
                  role="radio"
                  aria-checked={prefs()?.theme === t.value}
                  classList={{ on: prefs()?.theme === t.value }}
                  onClick={() => {
                    const p = prefs();
                    if (p) savePrefs({ ...p, theme: t.value });
                  }}
                >
                  {t.label}
                </span>
              )}
            </For>
          </span>
        </div>
      </div>
      <div class="prefs-foot">
        <span>Chats live in ~/QuickClaude</span>
        <span class="hint">⌘/ shortcuts</span>
      </div>
    </>
  );

  const keysView = (
    <>
      <div class="prefs-head">Keyboard Shortcuts</div>
      <div class="keys" ref={keysList}>
        <For each={KEY_SECTIONS}>
          {(section) => (
            <>
              <div class="keys-section">{section.title}</div>
              <div class="prefs">
                <For each={section.rows}>
                  {(row) => (
                    <div class="pref">
                      <span class="pref-text">
                        <span class="pref-title">{row.title}</span>
                      </span>
                      <span class="caps">
                        <For each={row.keys}>{(k) => <span>{k}</span>}</For>
                      </span>
                    </div>
                  )}
                </For>
              </div>
            </>
          )}
        </For>
      </div>
      <div class="prefs-foot">
        <span>esc close</span>
      </div>
    </>
  );

  const historyView = (
    <>
      {bar}
      <div class="history" ref={historyList}>
        <Show when={sessions()} fallback={<div class="status pad">Loading chats…</div>}>
          <For each={filtered()} fallback={<div class="status pad">No matching chats.</div>}>
            {(row, i) => (
              <button
                class="row"
                data-id={row.id}
                classList={{
                  selected: i() === selected(),
                  picked: picked().has(row.id),
                  armed: armed() && isTarget(row.id),
                }}
                onMouseMove={() => !picked().size && setSelected(i())}
                onClick={(e) => onRowClick(e, i(), row)}
              >
                <Show when={active().has(row.id)}>
                  <span class="lamp" title="Working" />
                </Show>
                <Show when={row.pinned}>
                  <svg class="pin" viewBox="0 0 16 16" aria-label="Pinned">
                    <path d="M10.5 1.5l4 4-2 1-2.5 2.5.5 3.5-1.5 1.5-3-3L2 15l-.5-.5 4-4-3-3L4 6l3.5.5L10 4z" />
                  </svg>
                </Show>
                <span class="title">{row.title}</span>
                <span class="when" classList={{ flash: !!pinFlash()?.ids.has(row.id) && !(armed() && isTarget(row.id)) }}>
                  {!(armed() && isTarget(row.id))
                    ? pinFlash()?.ids.has(row.id)
                      ? pinFlash()!.pinned
                        ? "Pinned"
                        : "Unpinned"
                      : ago(row.at)
                    : targets()[0]?.id !== row.id
                      ? "delete"
                      : targets().length > 1
                        ? `press ⌫ again to delete ${targets().length} chats`
                        : "press ⌫ again to delete"}
                </span>
              </button>
            )}
          </For>
        </Show>
      </div>
      <Show when={picked().size > 1}>{footer([`${picked().size} selected`])}</Show>
    </>
  );

  return (
    <div
      ref={panel}
      class="panel"
      classList={{ tall: tall(), full: full(), chat: mode() === "chat" && overlay() === "none" }}
      onClick={onClick}
      onMouseDown={onMouseDown}
    >
      <div class="top-ring" classList={{ on: onTop() }} />
      <Show when={overlay() !== "keys"} fallback={keysView}>
      <Show when={overlay() !== "prefs"} fallback={prefsView}>
      <Show when={overlay() !== "history"} fallback={historyView}>
        <Show
          when={mode() === "chat"}
          fallback={
            <>
              {bar}
              <>
                <Show when={lastAnswer()?.role === "assistant"}>
                  <div class="answer">
                    <div class="question">
                      {messages[last() - 1]?.images?.length ? "🖼 " : ""}
                      {messages[last() - 1]?.text}
                    </div>
                    <AssistantBody msg={lastAnswer()} index={last()} marks={marksFor(last())} />
                    {waiting()}
                  </div>
                </Show>
                <Show when={messages.length || notice()}>
                  <div class="footer">
                    <span class="notice">{notice()}</span>
                    <Show when={!busy() && messages.length}>
                      <span class="meta">{lastAnswer()?.meta}</span>
                    </Show>
                  </div>
                </Show>
              </>
            </>
          }
        >
          <div class="chat-head">
            <span class="chat-name">{chatName()}</span>
            <Show when={(context() ?? 0) >= CONTEXT_WARN}>
              <button
                class="context"
                classList={{ high: context()! >= CONTEXT_HIGH, armed: compactArmed() }}
                title={compactArmed() ? "Click again to compact this chat. Esc cancels." : "Context size of this chat. Click to compact it."}
                onClick={onChipClick}
              >
                <span class="grow">
                  <span>Compact&nbsp;</span>
                </span>
                {tokens(context()!)}
                <span class="grow">
                  <span>?</span>
                </span>
              </button>
            </Show>
          </div>
          <div class="list" ref={list}>
            <For each={messages}>
              {(m, i) => (
                <div class={`msg ${m.role}`}>
                  <Show when={m.role === "assistant"} fallback={<UserBody msg={m} />}>
                    <AssistantBody msg={m} index={i()} marks={marksFor(i())} />
                  </Show>
                </div>
              )}
            </For>
            {waiting()}
            <Show when={notice()}>
              <div class="error">{notice()}</div>
            </Show>
          </div>
          {bar}
        </Show>
      </Show>
      </Show>
      </Show>
    </div>
  );
}
