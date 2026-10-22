// Reels Studio editor. State lives on the server as immutable revisions; every edit commits a
// new revision (autosave) and the server returns the compiled timeline that both this preview
// and the final render use.
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
};
const fmt = (t) => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`; };
const clone = (o) => JSON.parse(JSON.stringify(o));
const uid = () => Math.random().toString(36).slice(2, 10);
const dbGain = (db) => Math.min(1, Math.pow(10, db / 20));

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body || opts.form ? "POST" : "GET"), headers: {} };
  if (opts.form) init.body = opts.form;
  else if (opts.body) { init.body = JSON.stringify(opts.body); init.headers["content-type"] = "application/json"; }
  if (opts.key) init.headers["idempotency-key"] = opts.key;
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(data.error || data.detail || r.statusText); e.status = r.status; e.data = data; throw e; }
  return data;
}

const S = {
  pid: null, project: null, tr: null, rev: null, comp: null, revisions: [], jobs: [],
  undo: [], redo: [], sel: null, anchor: null, open: new Set(), stage: "cut",
  assets: [], presets: [], fonts: [], queue: Promise.resolve(), playing: false, compT: 0,
};
const wordIndex = new Map();
const NOOP = Symbol("noop"); // returned by an edit() mutator to abandon the edit

// ------------------------------------------------------------------ banner / save state
function banner(msg, action) {
  const b = $("#banner");
  if (!msg) return b.classList.add("hidden");
  b.replaceChildren(msg, action ? h("button", { onclick: action.fn }, action.label) : "");
  b.classList.remove("hidden");
}
function saveState(s) {
  const e = $("#saveState");
  e.textContent = { saving: "Saving…", saved: "Saved", failed: "Save failed" }[s];
  e.classList.toggle("bad", s === "failed");
}

// ------------------------------------------------------------------ home
async function showHome() {
  stopPlayback();
  S.pid = null;
  history.replaceState(null, "", "/");
  $("#home").classList.remove("hidden");
  $("#editor").classList.add("hidden");
  for (const id of ["#title", "#stages", "#editTools"]) $(id).classList.add("hidden");
  banner(null);
  const list = await api("/api/projects");
  const box = $("#projectList");
  box.replaceChildren();
  if (!list.length) box.append(h("p", { class: "muted" }, "No projects yet."));
  for (const p of list) {
    const c = $("#tplProject").content.firstElementChild.cloneNode(true);
    c.querySelector("img").src = p.source ? `/api/projects/${p.id}/files/thumb` : "";
    c.querySelector("h3").textContent = p.title;
    const st = p.status === "ready" ? `${fmt(p.source.duration)} source · ${p.source.orientation}` :
      p.status === "failed" ? `Processing failed — open to retry` : p.status;
    c.querySelector("p").textContent = `${st} · ${new Date(p.updated * 1000).toLocaleString()}`;
    c.onclick = () => openProject(p.id);
    box.append(c);
  }
}

async function importFile(file) {
  const p = await api("/api/projects", { body: { title: "" } });
  const form = new FormData();
  form.append("file", file);
  banner(`Uploading ${file.name}…`);
  try {
    await api(`/api/projects/${p.id}/upload`, { form });
  } catch (e) {
    banner(`Could not import ${file.name}: ${e.message}`);
    await api(`/api/projects/${p.id}`, { method: "DELETE" });
    return;
  }
  banner(null);
  openProject(p.id);
}

// ------------------------------------------------------------------ project
async function openProject(pid) {
  S.pid = pid;
  history.replaceState(null, "", `/?p=${pid}`);
  const d = await api(`/api/projects/${pid}`);
  Object.assign(S, { project: d.project, tr: d.transcript, revisions: d.revisions, jobs: d.jobs });
  $("#home").classList.add("hidden");
  $("#editor").classList.remove("hidden");
  for (const id of ["#title", "#stages", "#editTools"]) $(id).classList.remove("hidden");
  $("#title").value = d.project.title;
  if (d.project.status !== "ready") return showProcessing(d);
  banner(null);
  S.rev = d.revision; S.comp = d.compiled; S.undo = []; S.redo = []; S.sel = null;
  wordIndex.clear();
  (S.tr?.words || []).forEach((w, i) => wordIndex.set(w.id, i));
  const v = $("#video");
  v.src = `/api/projects/${pid}/files/proxy`;
  v.onloadedmetadata = () => seekComp(0);
  [S.assets, S.presets, S.fonts] = await Promise.all([api("/api/assets"), api("/api/presets"), api("/api/fonts")]);
  if (S.tr?.note) banner(S.tr.note);
  renderAll();
}

function showProcessing(d) {
  const job = d.jobs.find((j) => j.kind === "process");
  $("#transcript").replaceChildren(h("div", { class: "notice" },
    d.project.status === "failed" || job?.status === "failed"
      ? [h("p", {}, `Processing failed: ${d.project.error || job?.error || "unknown error"}`),
         job ? h("button", { class: "primary", onclick: async () => { await api(`/api/jobs/${job.id}/retry`, { method: "POST" }); openProject(S.pid); } }, "Retry processing") : ""]
      : d.project.status === "empty" ? [h("p", {}, d.project.error || "No source yet."), h("button", { onclick: showHome }, "Back")]
      : [h("p", {}, `Processing: ${job?.stage || "queued"}…`), h("div", { class: "bar" }, h("i", { style: `width:${(job?.progress || 0) * 100}%` }))]));
  $("#inspector").replaceChildren();
  if (d.project.status === "processing") setTimeout(() => S.pid === d.project.id && openProject(S.pid), 1500);
}

// ------------------------------------------------------------------ committing edits
function edit(mutate, note = "") {
  const before = clone(S.rev);
  const next = clone(S.rev);
  if (mutate(next) === NOOP) return;
  S.undo.push(strip(before)); S.redo = [];
  commit(next, note);
}
const strip = (r) => { const { id, parent, actor, note, created, ...data } = r; return data; };

function commit(data, note) {
  saveState("saving");
  S.queue = S.queue.then(async () => {
    try {
      const out = await api(`/api/projects/${S.pid}/revisions`, { body: { base: S.rev.id, data: strip(data), note }, key: uid() });
      S.rev = out.revision; S.comp = out.compiled;
      S.revisions.push({ id: out.revision.id, actor: "user", note, created: out.revision.created });
      saveState("saved");
      renderAll();
    } catch (e) {
      saveState("failed");
      if (e.status === 409) banner("This project was changed in another tab or session.", { label: "Reload latest", fn: () => openProject(S.pid) });
      else banner(`Could not save: ${e.message}`, { label: "Retry", fn: () => { banner(null); commit(data, note); } });
    }
  });
}

function undo() {
  if (!S.undo.length) return;
  S.redo.push(strip(S.rev));
  commit(S.undo.pop(), "Undo");
}
function redo() {
  if (!S.redo.length) return;
  S.undo.push(strip(S.rev));
  commit(S.redo.pop(), "Redo");
}

function addDecision(r, op, ids, reason) {
  r.decisions.push({ op, ids, actor: "user", reason, at: Date.now() / 1000 });
}

// ------------------------------------------------------------------ selection helpers
function selIds() {
  if (!S.sel) return [];
  const [a, b] = S.sel;
  return S.tr.words.slice(Math.min(a, b), Math.max(a, b) + 1).map((w) => w.id);
}
function cutSel() {
  const ids = selIds().filter((id) => !S.rev.removed.includes(id));
  if (!ids.length) return;
  edit((r) => {
    r.removed = S.tr.words.map((w) => w.id).filter((id) => r.removed.includes(id) || ids.includes(id));
    ids.forEach((id) => (r.reasons[id] = "cut by you"));
    addDecision(r, "remove", ids, "manual cut");
  }, "Cut selection");
  S.sel = null;
}
function restoreIds(ids) {
  edit((r) => {
    const set = new Set(ids);
    if (!r.removed.some((id) => set.has(id))) return NOOP;
    r.removed = r.removed.filter((id) => !set.has(id));
    ids.forEach((id) => delete r.reasons[id]);
    addDecision(r, "restore", ids, "restored");
  }, "Restore");
}
function toggleEmph() {
  const ids = selIds().filter((id) => !S.rev.removed.includes(id));
  if (!ids.length) return;
  edit((r) => {
    const all = ids.every((id) => r.emphasis.includes(id));
    r.emphasis = all ? r.emphasis.filter((id) => !ids.includes(id)) : [...new Set([...r.emphasis, ...ids])];
  }, "Emphasis");
}
const wordText = (id) => S.rev.text[id] || S.tr.words[wordIndex.get(id)]?.w || "?";
function firstSelKept() { return selIds().find((id) => S.comp.words[id]); }
function lastSelKept() { return selIds().reverse().find((id) => S.comp.words[id]); }

// ------------------------------------------------------------------ transcript
function renderTranscript() {
  const box = $("#transcript");
  const words = S.tr?.words || [];
  if (!words.length) {
    box.replaceChildren(h("div", { class: "notice" }, "No speech transcript for this video. Captions and transcript cuts are unavailable; the full clip is used."));
    return;
  }
  const removed = new Set(S.rev.removed), emph = new Set(S.rev.emphasis);
  const anchors = new Set([...S.rev.overlays.map((o) => o.w0), ...S.rev.sounds.map((s) => s.w0)]);
  const sel = new Set(selIds());
  const frag = document.createDocumentFragment();
  const sentEnd = new Set((S.tr.sentences || []).map((s) => s.w1));
  let p = h("p"), i = 0;
  const groupAt = new Map(S.comp.groups.map((g) => [g.ids[0], g]));
  while (i < words.length) {
    const w = words[i];
    const g = groupAt.get(w.id);
    if (g) {
      const key = g.ids[0];
      const open = S.open.has(key);
      const chip = h("span", { class: "group" + (open ? " open" : ""), title: "Removed — click to review" });
      if (open) {
        chip.append(...g.ids.map((id) => wordSpan(words[wordIndex.get(id)], { cut: true, sel: sel.has(id) })),
          h("button", { class: "restore", onclick: (e) => { e.stopPropagation(); restoreIds(g.ids); } }, "Restore"),
          h("button", { class: "restore", onclick: (e) => { e.stopPropagation(); S.open.delete(key); renderTranscript(); } }, "Hide"));
      } else {
        chip.textContent = `⋯ ${(g.e - g.s).toFixed(1)}s removed · ${g.reason} @ ${fmt(g.s)}`;
        chip.onclick = () => { S.open.add(key); renderTranscript(); };
      }
      p.append(chip, " ");
      const last = g.ids[g.ids.length - 1];
      i = wordIndex.get(last) + 1;
      if (sentEnd.has(last)) { frag.append(p); p = h("p"); }
      continue;
    }
    const sp = wordSpan(w, { sel: sel.has(w.id), emph: emph.has(w.id), anchor: anchors.has(w.id) });
    p.append(sp, " ");
    if (sentEnd.has(w.id) || removed.has(w.id)) { frag.append(p); p = h("p"); }
    i++;
  }
  frag.append(p);
  box.replaceChildren(frag);
  renderSelbar();
}

function wordSpan(w, o) {
  const cls = ["w", o.sel && "sel", o.emph && "emph", o.anchor && "anchor", o.cut && "cutw", w.uncertain && "unc",
    S.rev.text[w.id] && "edited"].filter(Boolean).join(" ");
  const sp = h("span", { class: cls, "data-id": w.id, title: `${fmt(w.s)}${w.uncertain ? " · uncertain timing" : ""}` },
    S.rev.text[w.id] || w.w);
  return sp;
}

$("#transcript").addEventListener("click", (e) => {
  const sp = e.target.closest(".w");
  if (!sp) return;
  const i = wordIndex.get(sp.dataset.id);
  if (e.shiftKey && S.anchor != null) S.sel = [S.anchor, i];
  else { S.anchor = i; S.sel = [i, i]; }
  const w = S.tr.words[i];
  const t = S.comp.words[w.id];
  if (t) seekComp(t[0]);
  else { const v = $("#video"); v.currentTime = w.s; } // preview a removed take without restoring it
  renderTranscript();
  renderInspector();
});
$("#transcript").addEventListener("dblclick", (e) => {
  const sp = e.target.closest(".w");
  if (!sp) return;
  const id = sp.dataset.id;
  const cur = wordText(id);
  const txt = prompt("Correct caption text (the spoken audio is not changed):", cur);
  if (txt == null || txt === cur) return;
  edit((r) => { if (txt.trim()) r.text[id] = txt.trim(); else delete r.text[id]; }, "Correct text");
});

function renderSelbar() {
  const ids = selIds();
  $("#selInfo").textContent = ids.length ? `${ids.length} word${ids.length > 1 ? "s" : ""}: “${ids.slice(0, 6).map(wordText).join(" ")}${ids.length > 6 ? "…" : ""}”` : "No selection";
  const anyRemoved = ids.some((id) => S.rev.removed.includes(id));
  $("#selbar [data-act=cut]").disabled = !ids.some((id) => !S.rev.removed.includes(id));
  $("#selbar [data-act=restore]").disabled = !anyRemoved;
  $("#selbar [data-act=emph]").disabled = !ids.length;
}
$("#selbar").addEventListener("click", (e) => {
  const act = e.target.dataset?.act;
  if (act === "cut") cutSel();
  if (act === "restore") restoreIds(selIds());
  if (act === "emph") toggleEmph();
  if (act === "clear") { S.sel = null; renderTranscript(); renderInspector(); }
});

// ------------------------------------------------------------------ playback (skips removed ranges)
const video = $("#video");
function srcToComp(t) {
  for (const s of S.comp.segments) if (t >= s.s0 - 1e-3 && t <= s.s1 + 1e-3) return s.t0 + (t - s.s0);
  return null;
}
function compToSrc(t) {
  const segs = S.comp.segments;
  for (const s of segs) if (t < s.t0 + (s.s1 - s.s0)) return s.s0 + Math.max(0, t - s.t0);
  return segs.length ? segs[segs.length - 1].s1 : 0;
}
function seekComp(t) {
  S.compT = Math.max(0, Math.min(t, S.comp?.duration || 0));
  video.currentTime = compToSrc(S.compT);
  drawFrame();
}
function stopPlayback() { S.playing = false; video.pause(); pauseSounds(); $("#playBtn").textContent = "▶"; }
function togglePlay() {
  if (!S.comp) return;
  if (S.playing) return stopPlayback();
  if (S.compT >= S.comp.duration - 0.05) seekComp(0);
  S.playing = true;
  $("#playBtn").textContent = "❚❚";
  video.play();
  requestAnimationFrame(tick);
}
function tick() {
  if (!S.playing) return;
  const t = video.currentTime;
  let c = srcToComp(t);
  if (c == null) {
    const next = S.comp.segments.find((s) => s.s0 > t);
    if (!next) { S.compT = S.comp.duration; stopPlayback(); drawFrame(); return; }
    video.currentTime = next.s0;
    c = next.t0;
  } else {
    // jump the tiny gap between segments before removed frames show
    const seg = S.comp.segments.find((s) => t >= s.s0 - 1e-3 && t <= s.s1 + 1e-3);
    if (seg && t > seg.s1 - 0.02) {
      const next = S.comp.segments[S.comp.segments.indexOf(seg) + 1];
      if (!next) { S.compT = S.comp.duration; stopPlayback(); drawFrame(); return; }
      if (next.s0 > seg.s1 + 0.01) video.currentTime = next.s0;
    }
  }
  S.compT = c;
  drawFrame();
  syncSounds();
  requestAnimationFrame(tick);
}
$("#playBtn").onclick = togglePlay;

// ------------------------------------------------------------------ preview drawing
const GRADE_CSS = { none: "", warm: "sepia(.12) saturate(1.1)", cool: "hue-rotate(-8deg) saturate(.95) brightness(1.02)",
  punchy: "contrast(1.12) saturate(1.25)", bw: "grayscale(1) contrast(1.1)" };
const ovEls = new Map();

function drawFrame() {
  if (!S.comp) return;
  const t = S.compT;
  $("#clock").textContent = `${fmt(t)} / ${fmt(S.comp.duration)}`;
  const look = S.rev.look;
  video.style.filter = GRADE_CSS[look.grade] || "";
  video.style.transform = `scale(${look.zoom})`;
  drawCaptions(t);
  drawOverlays(t);
  drawTimeline();
  const cur = document.querySelector(".w.now");
  const active = S.comp.captions.find((c) => t >= c.t0 && t < c.t1)?.words.find((w) => t >= w.t0 && t < w.t1 + 0.05);
  if (cur && cur.dataset.id !== active?.id) cur.classList.remove("now");
  if (active) document.querySelector(`.w[data-id="${active.id}"]`)?.classList.add("now");
}

function drawCaptions(t) {
  const layer = $("#capLayer");
  const c = S.comp.captions.find((x) => t >= x.t0 && t < x.t1);
  if (!c) return layer.replaceChildren();
  const cap = S.rev.caption;
  const k = $("#frame").clientHeight / 1920;
  const box = h("div", { class: "cap" });
  box.style.top = `${cap.y * 100}%`;
  box.style.fontFamily = `"${cap.font}", Inter, sans-serif`;
  box.style.fontWeight = cap.weight;
  box.style.fontSize = `${cap.size * k}px`;
  const ow = cap.outline_w * k;
  let activeIdx = 0;
  c.words.forEach((w, i) => { if (t >= w.t0) activeIdx = i; });
  c.words.forEach((w, i) => {
    const sp = h("span", {}, w.text);
    sp.style.color = w.emph || i === activeIdx ? cap.hl : cap.color;
    if (w.emph) { sp.style.transform = "scale(1.15)"; if (cap.emph_italic) sp.style.fontStyle = "italic"; }
    if (cap.box) { sp.style.background = cap.box_color; sp.style.padding = `0 ${ow * 2}px`; }
    else if (ow) sp.style.textShadow = [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 1], [1, 0], [0, -1], [-1, 0]]
      .map(([x, y]) => `${x * ow}px ${y * ow}px 0 ${cap.outline}`).join(",");
    box.append(sp);
  });
  layer.replaceChildren(box);
}

const ANIM_D = 0.35;
function drawOverlays(t) {
  const layer = $("#ovLayer");
  const W = $("#frame").clientWidth, H = $("#frame").clientHeight;
  const live = new Set();
  for (const o of S.comp.overlays) {
    if (o.issue || o.t0 == null || t < o.t0 || t >= o.t1) continue;
    const a = S.assets.find((x) => x.id === o.asset);
    if (!a) continue;
    live.add(o.id);
    let el = ovEls.get(o.id);
    const src = `/api/assets/${a.id}`;
    if (!el || el.dataset.src !== src + o.mode) {
      el?.remove();
      el = a.type === "video" ? h("video", { muted: true, playsinline: true }) : h("img", { alt: "" });
      el.src = src; el.dataset.src = src + o.mode;
      layer.append(el); ovEls.set(o.id, el);
    }
    el.className = o.mode === "broll" ? "broll" : "";
    if (a.type === "video") {
      const want = o.src_in + (t - o.t0);
      if (Math.abs(el.currentTime - want) > 0.3) el.currentTime = want;
      if (S.playing && el.paused) el.play().catch(() => {});
      if (!S.playing) el.pause();
    }
    let x = 0, y = 0, rot = 0, op = 1;
    if (o.mode !== "broll") {
      const w = W * o.scale;
      el.style.width = `${w}px`;
      const hgt = a.width && a.height ? w * a.height / a.width : w;
      x = o.x * W - w / 2; y = o.y * H - hgt / 2;
      const pin = Math.min(1, (t - o.t0) / ANIM_D), pout = Math.max(0, (t - (o.t1 - ANIM_D)) / ANIM_D);
      if (o.anim_in === "slide-left") x += (W - x) * (1 - pin);
      if (o.anim_in === "slide-up") y += (H - y) * (1 - pin);
      if (o.anim_in === "fade" || o.anim_in === "pop") op *= pin;
      if (o.anim_in === "spin") rot += 360 * pin;
      if (o.anim_out === "slide-left") x -= (x + w) * pout;
      if (o.anim_out === "slide-up") y -= (y + hgt) * pout;
      if (o.anim_out === "fade" || o.anim_out === "pop") op *= 1 - pout;
      if (o.anim_out === "spin") rot += 360 * pout;
      el.style.left = `${x}px`; el.style.top = `${y}px`;
      el.style.transform = `rotate(${rot}deg)`;
    }
    el.style.opacity = op;
  }
  for (const [id, el] of ovEls) if (!live.has(id)) { el.remove(); ovEls.delete(id); }
}

// sounds in preview (approximate mix; the export uses the exact same placements)
const sndEls = new Map();
function syncSounds() {
  const t = S.compT;
  const speaking = S.comp.captions.some((c) => t >= c.t0 && t < c.t1);
  for (const s of S.comp.sounds) {
    if (s.t0 == null || s.t1 == null) continue;
    const a = S.assets.find((x) => x.id === s.asset);
    if (!a) continue;
    let el = sndEls.get(s.id);
    if (!el || el.dataset.asset !== a.id) { el?.pause(); el = new Audio(`/api/assets/${a.id}`); el.dataset.asset = a.id; sndEls.set(s.id, el); }
    const inside = t >= s.t0 && t < s.t1;
    if (inside) {
      let want = s.trim_in + (t - s.t0);
      if (s.loop && a.duration) want = s.trim_in + ((t - s.t0) % Math.max(0.05, a.duration - s.trim_in));
      if (el.paused) { el.currentTime = want; el.play().catch(() => {}); }
      el.volume = dbGain(s.gain_db) * (s.duck && speaking ? 0.3 : 1);
    } else if (!el.paused) el.pause();
  }
}
function pauseSounds() { for (const el of sndEls.values()) el.pause(); }

// ------------------------------------------------------------------ timeline
function drawTimeline() {
  const cv = $("#timeline");
  const W = cv.clientWidth, H = 96, dpr = window.devicePixelRatio || 1;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const D = Math.max(0.1, S.comp.duration), X = (t) => (t / D) * W;
  g.font = "10px Inter, sans-serif";
  const lane = (y, label) => { g.fillStyle = "#8b93a7"; g.fillText(label, 2, y + 9); };
  lane(2, "VIDEO");
  S.comp.segments.forEach((s, i) => {
    g.fillStyle = i % 2 ? "#33415e" : "#3d4f73";
    g.fillRect(X(s.t0), 14, Math.max(1, X(s.s1 - s.s0)), 18);
  });
  g.fillStyle = "#ffd40055";
  for (const c of S.comp.captions) g.fillRect(X(c.t0), 34, Math.max(1, X(c.t1 - c.t0) - 1), 4);
  lane(40, "VISUALS");
  for (const o of S.comp.overlays) {
    if (o.t0 == null) continue;
    g.fillStyle = o.issue ? "#ff6b6b" : o.mode === "broll" ? "#8a6eff" : "#6ea8ff";
    g.fillRect(X(o.t0), 52, Math.max(2, X((o.t1 ?? o.t0 + 0.5) - o.t0)), 12);
  }
  lane(66, "SOUND");
  for (const s of S.comp.sounds) {
    if (s.t0 == null) continue;
    g.fillStyle = s.issue ? "#ff6b6b" : "#4cd97b";
    g.fillRect(X(s.t0), 78, Math.max(2, X((s.t1 ?? s.t0 + 0.3) - s.t0)), 12);
  }
  g.fillStyle = "#fff";
  g.fillRect(X(S.compT) - 1, 0, 2, H);
}
$("#timeline").addEventListener("click", (e) => {
  if (!S.comp) return;
  const r = e.target.getBoundingClientRect();
  seekComp(((e.clientX - r.left) / r.width) * S.comp.duration);
  if (S.playing) syncSounds();
});
window.addEventListener("resize", () => S.comp && drawFrame());

// ------------------------------------------------------------------ inspector
const field = (label, input) => h("div", { class: "row" }, h("label", {}, label), input);
function num(val, min, max, step, on) {
  return h("input", { type: "range", min, max, step, value: val, onchange: (e) => on(parseFloat(e.target.value)) });
}
function color(val, on) { return h("input", { type: "color", value: val, onchange: (e) => on(e.target.value) }); }
function check(val, on) { return h("input", { type: "checkbox", checked: !!val, onchange: (e) => on(e.target.checked) }); }
function select(val, opts, on) {
  return h("select", { onchange: (e) => on(e.target.value) },
    opts.map(([v, l]) => h("option", { value: v, selected: v === val }, l)));
}

function renderInspector() {
  const box = $("#inspector");
  const fn = { cut: inspCut, visuals: inspVisuals, sound: inspSound, finish: inspFinish }[S.stage];
  box.replaceChildren(...fn());
}

function issuesList(kinds) {
  return S.comp.issues.filter((i) => !kinds || kinds.includes(i.kind)).map((i) => h("div", { class: i.kind.endsWith("warn") || i.kind === "word" ? "warn" : "issue" }, "⚠ " + i.msg));
}

function inspCut() {
  const src = S.project.source.duration, comp = S.comp.duration;
  const removedCount = S.comp.groups.length;
  return [
    h("div", { class: "stat" },
      h("div", {}, h("b", {}, fmt(comp)), h("span", {}, "reel length")),
      h("div", {}, h("b", {}, fmt(src)), h("span", {}, "source length")),
      h("div", {}, h("b", {}, String(removedCount)), h("span", {}, "removed sections")),
      h("div", {}, h("b", {}, String(S.comp.segments.length)), h("span", {}, "kept segments"))),
    h("h3", {}, "Rough cut"),
    h("p", { class: "muted" }, "The AI proposal removes fillers and earlier takes of repeated lines and tightens pauses. Removed sections stay in the transcript — expand one to review or restore it."),
    h("div", { class: "btns" },
      h("button", { onclick: rerunRoughCut }, "Re-run AI rough cut"),
      h("button", { onclick: () => { S.open = new Set(S.comp.groups.map((g) => g.ids[0])); renderTranscript(); } }, "Expand all removed"),
      h("button", { onclick: () => { S.open.clear(); renderTranscript(); } }, "Collapse all")),
    field("Max pause kept", num(S.rev.settings.max_gap, 0.1, 2, 0.05, (v) => edit((r) => (r.settings.max_gap = v), "Pause tightening"))),
    h("small", { class: "muted" }, `Pauses longer than ${S.rev.settings.max_gap.toFixed(2)}s are cut.`),
    field("Edge padding", num(S.rev.settings.pad, 0, 0.4, 0.02, (v) => edit((r) => (r.settings.pad = v), "Padding"))),
    h("h3", {}, "Selection"),
    S.sel ? h("div", { class: "btns" },
      h("button", { onclick: cutSel }, "Cut selection (Del)"),
      h("button", { onclick: () => restoreIds(selIds()) }, "Restore selection")) : h("p", { class: "muted" }, "Select words in the transcript to cut or restore them."),
    ...issuesList(),
    h("h3", {}, "History"),
    h("div", { class: "hist" }, [...S.revisions].reverse().slice(0, 60).map((r) =>
      h("div", {}, h("b", {}, `#${r.id}`), h("span", {}, `${r.actor} · ${r.note || "edit"}`),
        r.id !== S.rev.id ? h("button", { onclick: () => restoreRevision(r.id) }, "Restore") : h("small", { class: "ok" }, "current")))),
  ];
}

async function rerunRoughCut() {
  try {
    S.undo.push(strip(S.rev)); S.redo = [];
    const out = await api(`/api/projects/${S.pid}/roughcut`, { body: { base: S.rev.id } });
    S.rev = out.revision; S.comp = out.compiled;
    S.revisions.push({ id: out.revision.id, actor: "ai", note: "AI rough cut re-run" });
    renderAll();
  } catch (e) { banner(e.message); }
}
async function restoreRevision(n) {
  const r = await api(`/api/projects/${S.pid}/revisions/${n}`);
  S.undo.push(strip(S.rev)); S.redo = [];
  commit(strip(r), `Restored revision #${n}`);
}

function inspVisuals() {
  const cap = S.rev.caption, look = S.rev.look;
  const setCap = (k, note) => (v) => edit((r) => { r.caption[k] = v; r.caption_preset = null; }, note || "Caption style");
  const fonts = S.fonts.length ? S.fonts : ["Inter"];
  const capPresets = S.presets.filter((p) => p.category === "caption");
  const visualAssets = S.assets.filter((a) => ["image", "gif", "video"].includes(a.type));
  const applied = S.rev.caption_preset && S.presets.find((p) => p.id === S.rev.caption_preset.id);
  return [
    h("h3", {}, "Lookbook"),
    capPresets.length ? capPresets.map((p) => h("div", { class: "card" }, h("div", { class: "head" },
      h("b", {}, p.name), h("small", {}, `v${p.version}`),
      h("button", { onclick: () => applyPreset(p) }, "Apply"),
      h("button", { title: "Overwrite this preset with the current style", onclick: () => savePreset(p) }, "Update"),
      h("button", { class: "danger", "aria-label": "Delete preset", onclick: async () => { await api(`/api/presets/${p.id}`, { method: "DELETE" }); S.presets = await api("/api/presets"); renderInspector(); } }, "×"))))
      : h("p", { class: "muted" }, "No saved looks yet. Style the captions below, then save them as a look to reuse in any project."),
    applied ? h("small", { class: "muted" }, `Applied: ${applied.name} v${S.rev.caption_preset.version} (a snapshot — later preset changes don't alter this edit)`) : "",
    h("div", { class: "btns" }, h("button", { onclick: () => savePreset() }, "Save current style as look")),

    h("h3", {}, "Captions (whole reel)"),
    field("Show captions", check(S.rev.captions_on, (v) => edit((r) => (r.captions_on = v), "Captions on/off"))),
    field("Font", select(cap.font, fonts.map((f) => [f, f]), setCap("font"))),
    field("Weight", select(String(cap.weight), [["400", "Regular"], ["600", "Semibold"], ["800", "Extra bold"], ["900", "Black"]], (v) => setCap("weight")(+v))),
    field("Size", num(cap.size, 30, 160, 1, setCap("size"))),
    field("Text colour", color(cap.color, setCap("color"))),
    field("Highlight", color(cap.hl, setCap("hl"))),
    field("Outline", color(cap.outline, setCap("outline"))),
    field("Outline width", num(cap.outline_w, 0, 14, 0.5, setCap("outline_w"))),
    field("Box behind", check(cap.box, setCap("box"))),
    field("Box colour", color(cap.box_color, setCap("box_color"))),
    field("Vertical position", num(cap.y, 0.1, 0.9, 0.01, setCap("y"))),
    field("UPPERCASE", check(cap.upper, setCap("upper"))),
    field("Words per line", num(cap.words, 1, 8, 1, setCap("words"))),
    field("Italic emphasis", check(cap.emph_italic, setCap("emph_italic"))),
    h("div", { class: "btns" }, h("button", { disabled: !S.sel, onclick: toggleEmph }, "Emphasise selected words")),

    h("h3", {}, "Look"),
    field("Colour grade", select(look.grade, [["none", "None"], ["warm", "Warm"], ["cool", "Cool"], ["punchy", "Punchy"], ["bw", "Black & white"]], (v) => edit((r) => (r.look.grade = v), "Grade"))),
    field("Punch-in zoom", num(look.zoom, 1, 1.5, 0.01, (v) => edit((r) => (r.look.zoom = v), "Zoom"))),

    h("h3", {}, "Media library"),
    uploadBtn("Upload image / GIF / video", "image/*,video/mp4,video/quicktime,.gif"),
    h("div", { class: "lib" }, visualAssets.map((a) => h("div", { class: "item" },
      a.type === "video" ? h("video", { src: `/api/assets/${a.id}#t=0.5`, muted: true, preload: "metadata" }) : h("img", { src: `/api/assets/${a.id}`, alt: a.name }),
      h("div", { title: a.name }, a.name),
      h("button", { disabled: !S.sel, title: "Overlay on the selected words", onclick: () => addOverlay(a, "overlay") }, "Overlay"),
      a.type === "video" ? h("button", { disabled: !S.sel, title: "Full-frame B-roll over the selected words", onclick: () => addOverlay(a, "broll") }, "B-roll") : "",
    ))),
    S.sel ? "" : h("small", { class: "muted" }, "Select words in the transcript to place media on them."),

    h("h3", {}, `On this reel (${S.rev.overlays.length})`),
    ...S.comp.overlays.map(overlayCard),
  ];
}

function uploadBtn(label, accept) {
  const inp = h("input", { type: "file", accept, hidden: true, onchange: async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const form = new FormData();
    form.append("file", f);
    const rights = prompt("Usage-rights note for this file (optional):", "") || "";
    form.append("rights", rights);
    try { await api("/api/assets", { form }); S.assets = await api("/api/assets"); renderInspector(); }
    catch (err) { banner(`Upload failed: ${err.message}`); }
  } });
  return h("div", { class: "btns" }, h("button", { onclick: () => inp.click() }, label), inp);
}

function applyPreset(p) {
  edit((r) => {
    Object.assign(r.caption, p.params.caption || {});
    if (p.params.look) Object.assign(r.look, p.params.look);
    r.caption_preset = { id: p.id, version: p.version };
  }, `Applied look “${p.name}”`);
}
async function savePreset(existing) {
  const name = existing ? existing.name : prompt("Name this look:", "My caption look");
  if (!name) return;
  const p = await api("/api/presets", { body: { id: existing?.id, name, category: "caption", params: { caption: S.rev.caption, look: S.rev.look } } });
  S.presets = await api("/api/presets");
  edit((r) => (r.caption_preset = { id: p.id, version: p.version }), `Saved look “${name}”`);
}

function addOverlay(a, mode) {
  const w0 = firstSelKept(), w1 = lastSelKept();
  if (!w0) return banner("Select kept (not removed) words to anchor the media.");
  edit((r) => r.overlays.push({ id: uid(), asset: a.id, w0, w1, x: 0.5, y: 0.3, scale: 0.45, anim_in: mode === "broll" ? "none" : "pop",
    anim_out: mode === "broll" ? "none" : "fade", mode, src_in: 0, audio: false, note: "" }), mode === "broll" ? "Add B-roll" : "Add overlay");
}

function parseMotion(note) {
  // "slide in from the left, spin, and exit on 'money'" → animation plan
  const n = note.toLowerCase();
  const plan = {};
  const inPart = n.split(/exit|leave|out on|disappear/)[0];
  if (/slide.*(up|bottom)/.test(inPart)) plan.anim_in = "slide-up";
  else if (/slide/.test(inPart)) plan.anim_in = "slide-left";
  else if (/spin|rotate/.test(inPart)) plan.anim_in = "spin";
  else if (/fade/.test(inPart)) plan.anim_in = "fade";
  else if (/pop|bounce/.test(inPart)) plan.anim_in = "pop";
  const outPart = n.includes("exit") || n.includes("out") || n.includes("leave") ? n.slice(inPart.length) : "";
  if (outPart) plan.anim_out = /spin/.test(outPart) || (/spin/.test(n) && plan.anim_in !== "spin") ? "spin" : /slide/.test(outPart) ? "slide-left" : "fade";
  const m = note.match(/(?:exit|leave|out|disappear)\w*\s+(?:on|at|after)\s+(?:the\s+word\s+)?["“'‘]?([\w'’-]+)/i);
  if (m) plan.exitWord = m[1].replace(/[’']$/, "");
  return plan;
}

function overlayCard(o) {
  const a = S.assets.find((x) => x.id === o.asset);
  const set = (k, note) => (v) => edit((r) => { r.overlays.find((x) => x.id === o.id)[k] = v; }, note || "Overlay");
  const anims = [["none", "None"], ["fade", "Fade"], ["pop", "Pop"], ["slide-left", "Slide (side)"], ["slide-up", "Slide (up)"], ["spin", "Spin"]];
  const span = `“${wordText(o.w0)}” → “${wordText(o.w1 || o.w0)}”`;
  return h("div", { class: "card" + (o.issue ? " bad" : "") },
    h("div", { class: "head" }, h("b", {}, `${o.mode === "broll" ? "B-roll" : "Overlay"} · ${a ? a.name : "missing media"}`),
      h("button", { title: "Seek", onclick: () => o.t0 != null && seekComp(o.t0) }, "▶"),
      h("button", { class: "danger", onclick: () => edit((r) => (r.overlays = r.overlays.filter((x) => x.id !== o.id)), "Remove media") }, "Remove")),
    h("small", {}, `${span}${o.t0 != null ? ` · ${fmt(o.t0)}–${fmt(o.t1)}` : ""}`),
    o.issue ? h("div", { class: "issue" }, "⚠ " + o.issue) : "",
    h("div", { class: "btns" }, h("button", { disabled: !S.sel, onclick: () => edit((r) => { const x = r.overlays.find((y) => y.id === o.id); x.w0 = firstSelKept(); x.w1 = lastSelKept(); }, "Re-anchor") }, "Anchor to selection")),
    o.mode === "overlay" ? [
      field("X", num(o.x, 0, 1, 0.01, set("x"))), field("Y", num(o.y, 0, 1, 0.01, set("y"))),
      field("Size", num(o.scale, 0.05, 1, 0.01, set("scale"))),
      field("Enter", select(o.anim_in, anims, set("anim_in"))), field("Exit", select(o.anim_out, anims, set("anim_out"))),
      field("Motion note", h("input", { value: o.note, placeholder: "e.g. slide in, spin, exit on 'money'", onchange: (e) => applyMotion(o, e.target.value) })),
    ] : [
      field("Source in (s)", h("input", { type: "number", min: 0, step: 0.1, value: o.src_in, onchange: (e) => set("src_in", "B-roll in point")(parseFloat(e.target.value) || 0) })),
      a?.duration ? h("small", { class: "muted" }, `Clip is ${a.duration.toFixed(1)}s; placement is limited to what remains after the in point.`) : "",
      field("Use clip audio", check(o.audio, set("audio", "B-roll audio"))),
    ]);
}

function applyMotion(o, note) {
  const plan = parseMotion(note);
  edit((r) => {
    const x = r.overlays.find((y) => y.id === o.id);
    x.note = note;
    if (plan.anim_in) x.anim_in = plan.anim_in;
    if (plan.anim_out) x.anim_out = plan.anim_out;
    if (plan.exitWord) {
      const start = wordIndex.get(x.w0);
      const target = S.tr.words.slice(start).find((w) => !r.removed.includes(w.id) &&
        (r.text[w.id] || w.w).toLowerCase().replace(/[^\w'-]/g, "") === plan.exitWord.toLowerCase());
      if (target) x.w1 = target.id;
      else banner(`Couldn't find the word “${plan.exitWord}” after this overlay's start — exit timing unchanged.`);
    }
    if (!plan.anim_in && !plan.anim_out && !plan.exitWord) banner("That motion note wasn't understood. Try: slide in, spin, fade, pop, and “exit on <word>”.");
  }, "Motion note");
}

function inspSound() {
  const audio = S.assets.filter((a) => a.type === "audio");
  const st = S.rev.settings;
  return [
    h("h3", {}, "Narration"),
    field("Narration gain (dB)", num(st.narration_db, -12, 6, 0.5, (v) => edit((r) => (r.settings.narration_db = v), "Narration gain"))),
    h("h3", {}, "Sound library"),
    uploadBtn("Upload sound", "audio/*"),
    ...audio.map((a) => h("div", { class: "card" }, h("div", { class: "head" },
      h("b", {}, a.name), h("small", {}, `${(a.duration || 0).toFixed(1)}s`),
      h("button", { title: "Audition", onclick: () => audition(a) }, "▶"),
      h("button", { disabled: !S.sel, title: "Place at the first selected word", onclick: () => addSound(a, false) }, "At selection"),
      h("button", { title: "Music bed from the start, looped and ducked under speech", onclick: () => addSound(a, true) }, "As bed")),
      a.rights ? h("small", {}, a.rights) : "")),
    h("h3", {}, `On this reel (${S.rev.sounds.length})`),
    ...S.comp.sounds.map((s) => soundCard(s, audio)),
  ];
}
let auditionEl = null;
function audition(a) {
  if (auditionEl) { auditionEl.pause(); if (auditionEl.dataset.id === a.id) { auditionEl = null; return; } }
  auditionEl = new Audio(`/api/assets/${a.id}`); auditionEl.dataset.id = a.id; auditionEl.play();
}
function addSound(a, bed) {
  const w0 = bed ? null : firstSelKept();
  if (!bed && !w0) return banner("Select a kept word to place the sound on.");
  edit((r) => r.sounds.push({ id: uid(), asset: a.id, w0, gain_db: bed ? -14 : -4, trim_in: 0, length: null, loop: bed,
    fade_in: bed ? 0.5 : 0, fade_out: bed ? 1 : 0, duck: bed }), bed ? "Add music bed" : "Add sound");
}
function soundCard(s, audio) {
  const a = S.assets.find((x) => x.id === s.asset);
  const set = (k, note) => (v) => edit((r) => { r.sounds.find((x) => x.id === s.id)[k] = v; }, note || "Sound");
  return h("div", { class: "card" + (s.issue && !s.issue.startsWith("sound is shorter") ? " bad" : "") },
    h("div", { class: "head" }, h("b", {}, a ? a.name : "missing sound"),
      h("button", { onclick: () => s.t0 != null && seekComp(s.t0) }, "▶"),
      h("button", { class: "danger", onclick: () => edit((r) => (r.sounds = r.sounds.filter((x) => x.id !== s.id)), "Remove sound") }, "Remove")),
    h("small", {}, `${s.w0 ? `at “${wordText(s.w0)}”` : "from start"}${s.t0 != null && s.t1 != null ? ` · ${fmt(s.t0)}–${fmt(s.t1)}` : ""}`),
    s.issue ? h("div", { class: s.issue.startsWith("sound is shorter") ? "warn" : "issue" }, "⚠ " + s.issue) : "",
    field("Swap for", select(s.asset, audio.map((x) => [x.id, x.name]), set("asset", "Swap sound"))),
    field("Gain (dB)", num(s.gain_db, -30, 6, 0.5, set("gain_db"))),
    field("Trim start (s)", h("input", { type: "number", min: 0, step: 0.05, value: s.trim_in, onchange: (e) => set("trim_in", "Trim sound")(parseFloat(e.target.value) || 0) })),
    field("Length (s)", h("input", { type: "number", min: 0, step: 0.1, value: s.length ?? "", placeholder: "auto", onchange: (e) => set("length", "Sound length")(e.target.value === "" ? null : parseFloat(e.target.value)) })),
    field("Loop", check(s.loop, set("loop"))),
    field("Fade in (s)", num(s.fade_in, 0, 3, 0.1, set("fade_in"))),
    field("Fade out (s)", num(s.fade_out, 0, 3, 0.1, set("fade_out"))),
    field("Duck under voice", check(s.duck, set("duck"))),
    h("div", { class: "btns" }, h("button", { disabled: !S.sel, onclick: () => edit((r) => (r.sounds.find((x) => x.id === s.id).w0 = firstSelKept()), "Re-anchor sound") }, "Move to selection")));
}

function inspFinish() {
  const blocking = S.comp.issues.filter((i) => ["overlay", "sound", "composition"].includes(i.kind));
  const exports = S.jobs.filter((j) => j.kind === "export");
  return [
    h("h3", {}, "Export"),
    h("p", { class: "muted" }, `MP4 · 1080×1920 · H.264/AAC · ${S.rev.settings.fps} fps · ${fmt(S.comp.duration)} · revision #${S.rev.id}`),
    field("Frame rate", select(String(S.rev.settings.fps), [["24", "24"], ["25", "25"], ["30", "30"], ["60", "60"]], (v) => edit((r) => (r.settings.fps = +v), "Frame rate"))),
    blocking.length ? h("div", {}, h("div", { class: "issue" }, "Fix these before exporting:"), ...issuesList(["overlay", "sound", "composition"]))
      : h("div", { class: "ok" }, "✓ Ready to export"),
    ...issuesList(["word", "sound-warn"]),
    h("div", { class: "btns" }, h("button", { class: "primary", disabled: blocking.length > 0, onclick: startExport }, `Export revision #${S.rev.id}`)),
    h("h3", {}, "Exports"),
    exports.length ? exports.map(exportCard) : h("p", { class: "muted" }, "No exports yet."),
    h("h3", {}, "Coming next"),
    h("p", { class: "muted" }, "Editable CapCut handoff, frame capture, carousel/story and three trial reels are planned for R3."),
  ];
}
function exportCard(j) {
  const rev = j.params.revision;
  const outdated = rev !== S.rev.id;
  return h("div", { class: "card" + (j.status === "failed" ? " bad" : "") },
    h("div", { class: "head" }, h("b", {}, `Revision #${rev}`), h("small", {}, j.status),
      j.status === "ready" ? h("a", { href: `/api/projects/${S.pid}/exports/${j.result.file}`, download: "" }, h("button", {}, "Download")) : "",
      ["failed", "cancelled"].includes(j.status) ? h("button", { onclick: () => jobAction(j, "retry") }, "Retry") : "",
      ["queued", "working"].includes(j.status) ? h("button", { onclick: () => jobAction(j, "cancel") }, "Cancel") : ""),
    ["queued", "working"].includes(j.status) ? [h("div", { class: "bar" }, h("i", { style: `width:${(j.progress || 0) * 100}%` })), h("small", {}, j.stage)] : "",
    j.status === "ready" ? h("small", {}, `${fmt(j.result.duration)} · ${(j.result.size / 1e6).toFixed(1)} MB${outdated ? " · outdated: you've edited since" : ""}`) : "",
    j.error ? h("div", { class: "issue" }, j.error) : "",
    ...(j.result?.warnings || []).map((w) => h("div", { class: "warn" }, w)));
}
async function startExport() {
  const j = await api(`/api/projects/${S.pid}/export`, { body: { revision: S.rev.id } });
  S.jobs = [j, ...S.jobs.filter((x) => x.id !== j.id)];
  renderInspector();
  pollJobs();
}
async function jobAction(j, action) {
  const nj = await api(`/api/jobs/${j.id}/${action}`, { method: "POST" });
  S.jobs = S.jobs.map((x) => (x.id === nj.id ? nj : x));
  renderInspector();
  pollJobs();
}
let polling = false;
async function pollJobs() {
  if (polling) return;
  polling = true;
  while (S.pid && S.jobs.some((j) => ["queued", "working"].includes(j.status))) {
    await new Promise((r) => setTimeout(r, 1000));
    for (const j of S.jobs.filter((x) => ["queued", "working"].includes(x.status))) {
      const nj = await api(`/api/jobs/${j.id}`).catch(() => j);
      S.jobs = S.jobs.map((x) => (x.id === nj.id ? nj : x));
      if (nj.status === "ready" && j.status !== "ready") banner(`Export of revision #${nj.params.revision} is ready.`, { label: "Download", fn: () => (location.href = `/api/projects/${S.pid}/exports/${nj.result.file}`) });
    }
    if (S.stage === "finish") renderInspector();
  }
  polling = false;
}

// ------------------------------------------------------------------ wiring
function renderAll() {
  renderTranscript();
  renderInspector();
  drawFrame();
  $("#undoBtn").disabled = !S.undo.length;
  $("#redoBtn").disabled = !S.redo.length;
  if (S.jobs.some((j) => ["queued", "working"].includes(j.status))) pollJobs();
}

$("#stages").addEventListener("click", (e) => {
  const st = e.target.dataset?.stage;
  if (!st) return;
  S.stage = st;
  for (const b of $("#stages").children) b.classList.toggle("on", b.dataset.stage === st);
  renderInspector();
});
$("#homeBtn").onclick = showHome;
$("#undoBtn").onclick = undo;
$("#redoBtn").onclick = redo;
$("#title").addEventListener("change", (e) => api(`/api/projects/${S.pid}`, { method: "PATCH", body: { title: e.target.value } }));
$("#fileInput").addEventListener("change", (e) => e.target.files[0] && importFile(e.target.files[0]));
const drop = $("#drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && importFile(e.dataTransfer.files[0]); });

document.addEventListener("keydown", (e) => {
  if (!S.comp || e.target.matches("input, select, textarea")) return;
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if (mod && e.key.toLowerCase() === "y") { e.preventDefault(); redo(); }
  else if (e.key === " " && !e.target.matches("button")) { e.preventDefault(); togglePlay(); }
  else if ((e.key === "Delete" || e.key === "Backspace") && S.sel) { e.preventDefault(); cutSel(); }
  else if (e.key === "Escape") { S.sel = null; renderTranscript(); renderInspector(); }
});

const qp = new URLSearchParams(location.search).get("p");
qp ? openProject(qp).catch(showHome) : showHome();
