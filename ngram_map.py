#!/usr/bin/env python3
"""
ngram_map.py - a 3D map of how the words in your model connect.

    double-click ngram_map.py                 -> a small window: pick a model, click "Make map"
    python ngram_map.py model.db              -> same thing from the terminal
    python ngram_map.py model.db --words 800 --links 6 -o big_map.html

Writes one HTML file and opens it in your browser.
  * ball size     = how common the word is
  * colour        = a group of words that often appear next to each other
  * line          = the two words are often found side by side
  * moving dots   = reading direction (which word usually comes first)
Click a word to see what usually comes after it and before it.

How it works: words that often appear next to each other pull together like
springs, and every word pushes every other word away. Run that "physics"
a few hundred times and related words end up in the same neighbourhood.
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# three.js (the 3D library) is downloaded once into map_assets/ and then put INSIDE each map page,
# so the page works offline and can be shared as a single file.
THREE_VERSION = "0.147.0"         # last version with plain <script> files (no module loading needed)
THREE_FILES = {"three.min.js": "build/three.min.js",
               "OrbitControls.js": "examples/js/controls/OrbitControls.js"}
MIRRORS = [f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/", f"https://unpkg.com/three@{THREE_VERSION}/"]


def get_three(progress=print):
    """(three.js source, OrbitControls source), downloading them the first time. None if offline."""
    folder = HERE / "map_assets"
    out = []
    for name, rel in THREE_FILES.items():
        f = folder / f"{THREE_VERSION}-{name}"
        if not f.exists() or f.stat().st_size < 1000:
            progress("Downloading the 3D library (first time only)…")
            import urllib.request
            data = None
            for base in MIRRORS:
                try:
                    with urllib.request.urlopen(base + rel, timeout=30) as r:
                        data = r.read()
                    break
                except Exception:
                    continue
            if data is None:
                return None
            folder.mkdir(exist_ok=True)
            f.write_bytes(data)
        out.append(f.read_text(encoding="utf-8").replace("</script", "<\\/script"))
    return tuple(out)
PALETTE = ["#ff6b6b", "#4dabf7", "#69db7c", "#ffd43b", "#da77f2", "#ff922b",
           "#38d9a9", "#f783ac", "#a9e34b", "#748ffc", "#ffa8a8", "#66d9e8"]
OTHER = "#8a94a6"


# --------------------------------------------------------------------------
# 1. Read the model
# --------------------------------------------------------------------------
def load_graph(path, n_words, links_per_word, punct, skip_top):
    con = sqlite3.connect(path)
    levels = int(dict(con.execute("SELECT key, value FROM meta"))["levels"])
    vocab = con.execute("SELECT word, count FROM vocab WHERE id >= 2 ORDER BY id").fetchall()
    count = dict(vocab)
    freq_rank = {w: i + 1 for i, (w, _) in enumerate(vocab)}
    total = sum(count.values())

    def wanted(w):
        return punct or any(ch.isalnum() for ch in w)

    words = [w for w, _ in vocab if wanted(w)][skip_top:skip_top + n_words]
    idx = {w: i for i, w in enumerate(words)}

    nexts = {w: [] for w in words}       # what follows w
    prevs = {w: [] for w in words}       # what comes before w
    pair = {}                            # (i, j), i < j -> [count i->j, count j->i]
    for a, b, c, p in con.execute("SELECT context, next, count, prob FROM level_1"):
        if a in idx:
            nexts[a].append((b, p, c))
        if b in idx:
            prevs[b].append((a, p, c))
        if a in idx and b in idx and a != b:
            i, j = idx[a], idx[b]
            e = pair.setdefault((min(i, j), max(i, j)), [0, 0])
            e[0 if i < j else 1] += c
    con.close()

    # How strongly are two words associated? "Dice score": how often they sit
    # side by side, compared with how common each word is on its own. Plain
    # counts would connect everything to "the"; this rewards real pairs.
    best = {i: [] for i in range(len(words))}
    for (i, j), (cij, cji) in pair.items():
        dice = 2 * (cij + cji) / (count[words[i]] + count[words[j]])
        best[i].append((dice, j))
        best[j].append((dice, i))
    edges = {}
    for i, lst in best.items():
        for dice, j in sorted(lst, reverse=True)[:links_per_word]:
            key = (min(i, j), max(i, j))
            edges[key] = (dice, *pair[key])

    # drop words that ended up with no lines at all
    linked = sorted({i for e in edges for i in e})
    remap = {old: new for new, old in enumerate(linked)}
    words = [words[i] for i in linked]
    edge_list = [(remap[i], remap[j], d, cij, cji) for (i, j), (d, cij, cji) in edges.items()]

    nodes = []
    for w in words:
        nodes.append({
            "w": w, "c": count[w], "r": freq_rank[w],
            "next": [[b, round(p, 4), c] for b, p, c in sorted(nexts[w], key=lambda x: -x[1])[:12]],
            "prev": [[a, round(p, 4), c] for a, p, c in sorted(prevs[w], key=lambda x: -x[2])[:12]],
        })
    return nodes, edge_list, {"total": total, "levels": levels}


# --------------------------------------------------------------------------
# 2. Lay it out in 3D (force-directed: springs + repulsion)
# --------------------------------------------------------------------------
def layout(n, edges, iters=350, seed=7, progress=None):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)) * 3
    src = np.array([e[0] for e in edges])
    dst = np.array([e[1] for e in edges])
    w = np.array([e[2] for e in edges])
    w = np.sqrt(w / w.max()) + 0.15              # spring strength
    k = 1.0                                      # ideal distance
    step = 1.0
    for it in range(iters):
        if progress and it % 50 == 0:
            progress(f"Arranging {n} words in 3D… {100 * it // iters}%")
        delta = pos[:, None, :] - pos[None, :, :]
        d2 = (delta ** 2).sum(-1) + 0.01
        force = (delta / d2[..., None]).sum(1) * k * k          # everyone pushes apart
        diff = pos[src] - pos[dst]
        dist = np.sqrt((diff ** 2).sum(1)) + 1e-9
        pull = diff * (dist * w / k)[:, None]                   # springs pull together
        np.add.at(force, src, -pull)
        np.add.at(force, dst, pull)
        force -= 0.02 * pos * np.linalg.norm(pos, axis=1, keepdims=True)   # keep it compact
        length = np.linalg.norm(force, axis=1, keepdims=True) + 1e-9
        pos += force / length * np.minimum(length, step)
        step = max(0.02, step * 0.985)
    pos -= pos.mean(0)
    pos *= 220 / np.percentile(np.linalg.norm(pos, axis=1), 95)
    return pos


# --------------------------------------------------------------------------
# 3. Find groups of words that stick together (label propagation)
# --------------------------------------------------------------------------
def find_groups(n, edges, rounds=30, seed=3):
    rng = np.random.default_rng(seed)
    nbrs = [[] for _ in range(n)]
    for i, j, d, *_ in edges:
        nbrs[i].append((j, d))
        nbrs[j].append((i, d))
    label = list(range(n))
    for _ in range(rounds):
        changed = 0
        for i in rng.permutation(n):
            score = {}
            for j, d in nbrs[i]:
                score[label[j]] = score.get(label[j], 0) + d
            if score:
                top = max(score.items(), key=lambda x: (x[1], -x[0]))[0]
                if top != label[i]:
                    label[i] = top
                    changed += 1
        if not changed:
            break
    sizes = {}
    for g in label:
        sizes[g] = sizes.get(g, 0) + 1
    order = sorted(sizes, key=lambda g: -sizes[g])
    gid = {g: (k if k < len(PALETTE) and sizes[g] >= 3 else -1) for k, g in enumerate(order)}
    return [gid[g] for g in label]


# --------------------------------------------------------------------------
# 4. Write the page
# --------------------------------------------------------------------------
def build_page(model_path, nodes, edges, info, libs, progress=None):
    pos = layout(len(nodes), edges, progress=progress)
    groups = find_groups(len(nodes), edges)
    legend = []
    for g in range(len(PALETTE)):
        members = [n for n, gg in zip(nodes, groups) if gg == g]
        if members:
            members.sort(key=lambda n: -n["c"])
            legend.append({"color": PALETTE[g], "size": len(members),
                           "name": ", ".join(m["w"] for m in members[:3])})
    for n, p, g in zip(nodes, pos, groups):
        n["p"] = [round(float(x), 2) for x in p]
        n["g"] = PALETTE[g] if g >= 0 else OTHER
    data = {
        "model": os.path.basename(model_path), "total": info["total"],
        "nodes": nodes,
        "links": [[i, j, round(d, 5), cij, cji] for i, j, d, cij, cji in edges],
        "legend": legend,
    }
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    if libs:      # library inside the page: works offline
        lib_tags = f"<script>{libs[0]}</script>\n<script>{libs[1]}</script>"
    else:         # couldn't download: let the browser fetch it
        lib_tags = "\n".join(f'<script src="{MIRRORS[0]}{rel}"></script>' for rel in THREE_FILES.values())
    page = PAGE.replace("__TITLE__", f"Word map · {os.path.basename(model_path)}")
    page = page.replace("__DATA__", js)
    return page.replace("__LIBS__", lib_tags)


def make_map(model_path, out_path=None, words=500, links=5, punct=False, skip_top=0, progress=print):
    """Read a model, lay it out, write the page. Returns (output path, words placed, links)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    out = Path(out_path or Path(model_path).with_name(Path(model_path).stem + "_map.html")).resolve()
    libs = get_three(progress)
    progress("Reading the model…")
    nodes, edges, info = load_graph(model_path, words, links, punct, skip_top)
    if len(nodes) < 2:
        raise ValueError("Not enough linked words to draw. Try a bigger model or more words.")
    html = build_page(model_path, nodes, edges, info, libs, progress)
    out.write_text(html, encoding="utf-8")
    progress(f"Saved {out.name}: {len(nodes)} words, {len(edges)} links"
             + ("" if libs else "  (needs internet to show: library download failed)"))
    return out, len(nodes), len(edges)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0b0e14; --panel:rgba(18,22,32,.88); --line:rgba(255,255,255,.08); --text:#e6e9ef; --dim:#8a94a6; --accent:#ffd43b; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; overflow:hidden; background:var(--bg); color:var(--text);
              font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  canvas { display:block; }
  .card { position:absolute; background:var(--panel); border:1px solid var(--line); border-radius:12px;
          backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }
  #top { top:14px; left:14px; padding:12px 14px; width:300px; }
  #top h1 { font-size:15px; margin:0 0 8px; font-weight:600; }
  #top h1 span { color:var(--dim); font-weight:400; }
  #search { width:100%; padding:8px 10px; border-radius:8px; border:1px solid var(--line);
            background:rgba(255,255,255,.06); color:var(--text); font-size:14px; outline:none; }
  #search:focus { border-color:var(--accent); }
  .opts { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:10px; color:var(--dim); font-size:13px; }
  .opts label { cursor:pointer; user-select:none; }
  .opts select { background:#1a1f2b; color:var(--text); border:1px solid var(--line); border-radius:6px; }
  #legend { left:14px; bottom:14px; padding:10px 14px; max-width:300px; font-size:12.5px; }
  #legend .row { display:flex; align-items:center; gap:8px; margin:3px 0; color:var(--dim); cursor:pointer; }
  #legend .row:hover { color:var(--text); }
  #legend .dot { width:10px; height:10px; border-radius:50%; flex:none; }
  #legend .help { margin-top:8px; padding-top:8px; border-top:1px solid var(--line); color:var(--dim); }
  #panel { top:14px; right:14px; bottom:14px; width:330px; padding:16px 18px; overflow-y:auto;
           transform:translateX(360px); transition:transform .25s ease; }
  #panel.open { transform:none; }
  #panel h2 { margin:0; font-size:28px; font-weight:650; }
  #panel .sub { color:var(--dim); margin:2px 0 14px; }
  #panel h3 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim);
              margin:16px 0 6px; font-weight:600; }
  .bar { display:grid; grid-template-columns:96px 1fr 52px; align-items:center; gap:8px; margin:3px 0; font-size:13.5px; }
  .bar .w { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .bar .w.link { cursor:pointer; text-decoration:underline dotted rgba(255,255,255,.35); text-underline-offset:3px; }
  .bar .w.link:hover { color:var(--accent); }
  .bar .w.off { color:var(--dim); }
  .bar .track { height:7px; background:rgba(255,255,255,.07); border-radius:4px; overflow:hidden; }
  .bar .fill { display:block; height:100%; border-radius:4px; }
  .bar .p { text-align:right; color:var(--dim); font-variant-numeric:tabular-nums; font-size:12.5px; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { padding:3px 9px; border-radius:999px; background:rgba(255,255,255,.07); cursor:pointer; font-size:13px; }
  .chip:hover { background:rgba(255,212,59,.2); }
  #close { position:absolute; top:12px; right:14px; cursor:pointer; color:var(--dim); font-size:20px; background:none; border:0; }
  #close:hover { color:var(--text); }
  #tip { position:absolute; pointer-events:none; padding:5px 9px; border-radius:7px; background:rgba(0,0,0,.8);
         font-size:13px; display:none; white-space:nowrap; }
  .note { color:var(--dim); font-size:12.5px; margin-top:6px; }
</style>
</head>
<body>
<div id="top" class="card">
  <h1>Word map <span id="meta"></span></h1>
  <input id="search" list="wordlist" placeholder="Find a word…  (Enter)" autocomplete="off">
  <datalist id="wordlist"></datalist>
  <div class="opts">
    <label>Labels <select id="labels"><option value="all">all</option><option value="big" selected>common</option><option value="none">none</option></select></label>
    <label><input type="checkbox" id="showLinks" checked> lines</label>
    <label><input type="checkbox" id="spin" checked> rotate</label>
  </div>
</div>
<div id="legend" class="card"></div>
<div id="panel" class="card"><button id="close" title="Close (Esc)">×</button><div id="pbody"></div></div>
<div id="tip"></div>

__LIBS__
<script>
(function () {
if (!window.THREE || !THREE.OrbitControls) {
  document.body.innerHTML = '<div style="padding:40px;font:16px system-ui;color:#e6e9ef">' +
    'Couldn\'t load the 3D library. Connect to the internet once and make the map again.</div>';
  return;
}
const OrbitControls = THREE.OrbitControls;

const DATA = __DATA__;
const tip = document.getElementById('tip');
const N = DATA.nodes, L = DATA.links;
const byWord = new Map(N.map((n, i) => [n.w, i]));
const nbrs = N.map(() => []);
L.forEach(([i, j, d, cij, cji], k) => { nbrs[i].push([j, d, k]); nbrs[j].push([i, d, k]); });
document.getElementById('meta').textContent = `· ${DATA.model} · ${N.length} words`;

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);
scene.fog = new THREE.FogExp2(0x0b0e14, 0.0011);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 1, 5000);
camera.position.set(0, 60, 520);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.35;
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const sun = new THREE.DirectionalLight(0xffffff, 0.75);
sun.position.set(200, 300, 400);
scene.add(sun);

// ---------- nodes ----------
const maxC = Math.max(...N.map(n => n.c));
const radius = n => 1.1 + 4.6 * Math.sqrt(Math.log(1 + n.c) / Math.log(1 + maxC));
const sphere = new THREE.SphereGeometry(1, 20, 14);
const meshes = N.map((n, i) => {
  const col = new THREE.Color(n.g);
  const m = new THREE.Mesh(sphere, new THREE.MeshStandardMaterial({
    color: col, emissive: col, emissiveIntensity: 0.25, roughness: 0.6, transparent: true }));
  m.position.set(...n.p);
  m.scale.setScalar(radius(n));
  m.userData.i = i;
  scene.add(m);
  return m;
});

// labels: text drawn on a canvas, shown as a sprite that always faces the camera
function makeLabel(text, color) {
  const fs = 44, pad = 10, c = document.createElement('canvas'), g = c.getContext('2d');
  g.font = `600 ${fs}px system-ui, "Segoe UI", sans-serif`;
  c.width = Math.ceil(g.measureText(text).width) + pad * 2; c.height = fs + pad * 2;
  g.font = `600 ${fs}px system-ui, "Segoe UI", sans-serif`;
  g.textBaseline = 'middle';
  g.lineWidth = 8; g.strokeStyle = 'rgba(0,0,0,.85)'; g.strokeText(text, pad, c.height / 2);
  g.fillStyle = color; g.fillText(text, pad, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 4;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  s.userData.aspect = c.width / c.height;
  return s;
}
const labels = N.map((n, i) => {
  const s = makeLabel(n.w, '#f1f3f7');
  const h = 4 + radius(n) * 0.9;
  s.scale.set(h * s.userData.aspect, h, 1);
  s.position.set(n.p[0], n.p[1] + radius(n) + h * 0.55, n.p[2]);
  s.renderOrder = 2;
  scene.add(s);
  return s;
});
const bigCut = [...N].map(n => n.c).sort((a, b) => b - a)[Math.min(N.length - 1, 120)];

// ---------- links ----------
const maxD = Math.max(...L.map(l => l[2]));
function lineGeometry(list, strength) {
  const pos = new Float32Array(list.length * 6), col = new Float32Array(list.length * 6);
  list.forEach(([i, j, d], k) => {
    pos.set(N[i].p, k * 6); pos.set(N[j].p, k * 6 + 3);
    const a = new THREE.Color(N[i].g), b = new THREE.Color(N[j].g);
    const s = strength(d);
    col.set([a.r * s, a.g * s, a.b * s], k * 6); col.set([b.r * s, b.g * s, b.b * s], k * 6 + 3);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return geo;
}
const baseLines = new THREE.LineSegments(
  lineGeometry(L, d => 0.25 + 0.75 * Math.sqrt(d / maxD)),
  new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4, depthWrite: false }));
scene.add(baseLines);
const hiLines = new THREE.LineSegments(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false }));
scene.add(hiLines);

// moving dots show reading direction on the selected word's lines
let dotTex = null;
function dotTexture() {           // a soft round dot instead of a square
  if (dotTex) return dotTex;
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'), grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(0.5, 'rgba(255,255,255,.9)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
  return (dotTex = new THREE.CanvasTexture(c));
}
const flow = { pts: null, paths: [] };
function setFlow(i) {
  if (flow.pts) { scene.remove(flow.pts); flow.pts.geometry.dispose(); flow.pts = null; }
  flow.paths = [];
  if (i == null) return;
  for (const [j, d, k] of nbrs[i]) {
    const [a, b, , cab, cba] = L[k];           // cab = times "a b", cba = times "b a"
    const fwd = cab >= cba ? [a, b] : [b, a];
    const n = 2 + Math.round(4 * Math.sqrt(d / maxD));
    for (let t = 0; t < n; t++) flow.paths.push([new THREE.Vector3(...N[fwd[0]].p), new THREE.Vector3(...N[fwd[1]].p), t / n]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(flow.paths.length * 3), 3));
  flow.pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xffd43b, size: 3.6, map: dotTexture(),
    transparent: true, alphaTest: 0.2, depthWrite: false }));
  flow.pts.renderOrder = 3;
  scene.add(flow.pts);
}

// ---------- highlighting ----------
let hover = null, selected = null;
function applyFocus() {
  const f = selected ?? hover;
  const near = new Set(f == null ? [] : [f, ...nbrs[f].map(x => x[0])]);
  const mode = document.getElementById('labels').value;
  meshes.forEach((m, i) => {
    const on = f == null || near.has(i);
    m.material.opacity = on ? 1 : 0.07;
    m.material.depthWrite = on;
    m.material.emissiveIntensity = i === f ? 0.8 : 0.25;
  });
  // focused word + its neighbours always get labels; otherwise follow the Labels setting
  labels.forEach((s, i) => {
    s.visible = f != null ? near.has(i) : (mode === 'all' || (mode === 'big' && N[i].c >= bigCut));
  });
  baseLines.visible = document.getElementById('showLinks').checked;
  baseLines.material.opacity = f == null ? 0.4 : 0.06;
  const mine = f == null ? [] : nbrs[f].map(([, , k]) => L[k]);
  hiLines.geometry.dispose();
  hiLines.geometry = lineGeometry(mine, () => 1.0);
}

// ---------- side panel ----------
const panel = document.getElementById('panel'), pbody = document.getElementById('pbody');
const esc = s => s.replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
function bars(list, color, label) {
  if (!list.length) return '<div class="note">Nothing stored.</div>';
  const top = Math.max(...list.map(x => x[1]));
  return list.map(([w, p, c]) => {
    const inMap = byWord.has(w);
    return `<div class="bar" title="${esc(label(w, p, c))}">
      <span class="w ${inMap ? 'link' : 'off'}" data-w="${esc(w)}">${esc(w)}</span>
      <span class="track"><span class="fill" style="width:${(100 * p / top).toFixed(1)}%;background:${color}"></span></span>
      <span class="p">${p < 0.01 ? p.toFixed(3) : p.toFixed(2)}</span></div>`;
  }).join('');
}
function showPanel(i) {
  const n = N[i];
  const conn = [...nbrs[i]].sort((a, b) => b[1] - a[1]);
  pbody.innerHTML = `
    <h2 style="color:${n.g}">${esc(n.w)}</h2>
    <div class="sub">seen ${n.c.toLocaleString()} times · #${n.r} most common word</div>
    <h3>Usually comes next  ·  P(next | ${esc(n.w)})</h3>
    ${bars(n.next, n.g, (w, p, c) => `"${n.w} ${w}" seen ${c}x. ${(100 * p).toFixed(1)}% of the time "${n.w}" is followed by "${w}"`)}
    <h3>Usually comes before  ·  P(${esc(n.w)} | before)</h3>
    ${bars(n.prev, '#8a94a6', (w, p, c) => `"${w} ${n.w}" seen ${c}x. ${(100 * p).toFixed(1)}% of the time "${w}" is followed by "${n.w}"`)}
    <div class="note">Bars show likelihood. Underlined words are on the map; click to jump.</div>
    <h3>Linked on the map</h3>
    <div class="chips">${conn.map(([j]) => `<span class="chip" data-w="${esc(N[j].w)}">${esc(N[j].w)}</span>`).join('')}</div>`;
  panel.classList.add('open');
}
pbody.addEventListener('click', e => {
  const w = e.target.closest('[data-w]')?.dataset.w;
  if (w && byWord.has(w)) select(byWord.get(w));
});

// ---------- camera fly-to ----------
let fly = null;
function flyTo(i) {
  const target = new THREE.Vector3(...N[i].p);
  const dir = camera.position.clone().sub(controls.target).normalize();
  fly = { t: 0, fromPos: camera.position.clone(), fromTgt: controls.target.clone(),
          toPos: target.clone().add(dir.multiplyScalar(140)), toTgt: target };
}

function select(i) {
  selected = i;
  tip.style.display = 'none';
  if (i == null) { panel.classList.remove('open'); setFlow(null); applyFocus(); return; }
  showPanel(i); setFlow(i); applyFocus(); flyTo(i);
  document.getElementById('spin').checked = false; controls.autoRotate = false;
}
window.wordMap = { select: w => select(typeof w === 'string' ? byWord.get(w) : w) };

// ---------- mouse ----------
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
let downAt = null;
function pick(ev) {
  mouse.set(ev.clientX / innerWidth * 2 - 1, -ev.clientY / innerHeight * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(meshes, false).find(h => h.object.material.opacity > 0.5 || selected == null);
  return hit ? hit.object.userData.i : null;
}
renderer.domElement.addEventListener('pointermove', ev => {
  const i = pick(ev);
  if (i !== hover) { hover = i; if (selected == null) applyFocus(); }
  renderer.domElement.style.cursor = i == null ? 'grab' : 'pointer';
  if (i == null) { tip.style.display = 'none'; return; }
  tip.style.display = 'block';
  tip.style.left = ev.clientX + 14 + 'px'; tip.style.top = ev.clientY + 10 + 'px';
  tip.textContent = `${N[i].w} · seen ${N[i].c.toLocaleString()}x`;
});
renderer.domElement.addEventListener('pointerdown', ev => downAt = [ev.clientX, ev.clientY]);
renderer.domElement.addEventListener('pointerup', ev => {
  if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 5) return;
  select(pick(ev));
});
renderer.domElement.addEventListener('pointerleave', () => tip.style.display = 'none');
addEventListener('keydown', e => { if (e.key === 'Escape') select(null); });
document.getElementById('close').onclick = () => select(null);

// ---------- UI ----------
const dl = document.getElementById('wordlist');
dl.innerHTML = [...N].sort((a, b) => a.w.localeCompare(b.w)).map(n => `<option value="${esc(n.w)}">`).join('');
const search = document.getElementById('search');
search.addEventListener('change', go);
search.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
function go() {
  const w = search.value.trim().toLowerCase();
  if (byWord.has(w)) select(byWord.get(w));
  else if (w) { search.style.borderColor = '#ff6b6b'; setTimeout(() => search.style.borderColor = '', 900); }
}
document.getElementById('labels').onchange = applyFocus;
document.getElementById('showLinks').onchange = applyFocus;
document.getElementById('spin').onchange = e => controls.autoRotate = e.target.checked;

const leg = document.getElementById('legend');
leg.innerHTML = DATA.legend.map((g, k) =>
  `<div class="row" data-g="${k}"><span class="dot" style="background:${g.color}"></span>${esc(g.name)} <span style="opacity:.6">(${g.size})</span></div>`).join('') +
  `<div class="help">Size = how common · Colour = words that stick together · Line = often side by side ·
   Moving dots = reading direction<br>Drag to rotate · scroll to zoom · right-drag to pan · click a word</div>`;
leg.addEventListener('click', e => {
  const k = e.target.closest('.row')?.dataset.g;
  if (k == null) return;
  const col = DATA.legend[k].color;
  const members = N.map((n, i) => n.g === col ? i : -1).filter(i => i >= 0);
  const i = members.reduce((a, b) => N[a].c >= N[b].c ? a : b);
  select(i);
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------- loop ----------
let last = performance.now();
applyFocus();
renderer.setAnimationLoop(() => {
  const now = performance.now(), dt = Math.min((now - last) / 1000, 0.05); last = now;
  if (fly) {
    fly.t = Math.min(1, fly.t + dt / 0.9);
    const e = 1 - Math.pow(1 - fly.t, 3);
    camera.position.lerpVectors(fly.fromPos, fly.toPos, e);
    controls.target.lerpVectors(fly.fromTgt, fly.toTgt, e);
    if (fly.t >= 1) fly = null;
  }
  if (flow.pts) {
    const arr = flow.pts.geometry.attributes.position.array;
    flow.paths.forEach((p, k) => {
      p[2] = (p[2] + dt * 0.45) % 1;
      const v = p[0].clone().lerp(p[1], p[2]);
      arr[k * 3] = v.x; arr[k * 3 + 1] = v.y; arr[k * 3 + 2] = v.z;
    });
    flow.pts.geometry.attributes.position.needsUpdate = true;
  }
  controls.update();
  renderer.render(scene, camera);
});
})();
</script>
</body>
</html>
"""


def window(initial_model=None):
    """A small window, for when the script is double-clicked."""
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    root.title("Word map maker")
    root.resizable(False, False)
    f = ttk.Frame(root, padding=14)
    f.pack(fill="both", expand=True)

    dbs = sorted(HERE.glob("*.db"), key=lambda p: -p.stat().st_mtime)
    model = tk.StringVar(value=initial_model or (str(dbs[0]) if dbs else ""))
    ttk.Label(f, text="Model:").grid(row=0, column=0, sticky="w")
    box = ttk.Combobox(f, textvariable=model, values=[str(p) for p in dbs], width=55)
    box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
    ttk.Button(f, text="Browse…", command=lambda: model.set(
        filedialog.askopenfilename(filetypes=[("Model", "*.db")]) or model.get())).grid(row=0, column=4)

    v_words, v_links, v_skip, v_punct = tk.IntVar(value=500), tk.IntVar(value=5), tk.IntVar(value=0), tk.BooleanVar()
    rows = [("Words on the map", v_words, 50, 1500, "more = slower to arrange"),
            ("Links per word", v_links, 1, 15, "strongest connections kept"),
            ("Hide most common", v_skip, 0, 300, "e.g. 20 hides the, of, and…")]
    for r, (label, var, lo, hi, hint) in enumerate(rows, start=1):
        ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Spinbox(f, from_=lo, to=hi, textvariable=var, width=7).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Label(f, text=hint, foreground="#777").grid(row=r, column=2, columnspan=3, sticky="w")
    ttk.Checkbutton(f, text="Include punctuation (. , ! ?) as words", variable=v_punct).grid(
        row=4, column=0, columnspan=3, sticky="w", pady=(4, 8))

    status = tk.StringVar(value="Pick a model and click Make map.")
    bar = ttk.Progressbar(f, mode="indeterminate")
    last = {"path": None}
    btn = ttk.Button(f, text="Make map and open it")
    open_btn = ttk.Button(f, text="Open last map", state="disabled",
                          command=lambda: webbrowser.open(last["path"].as_uri()))
    btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
    open_btn.grid(row=5, column=2, sticky="w", padx=6, pady=(4, 0))
    bar.grid(row=6, column=0, columnspan=5, sticky="ew", pady=(10, 4))
    ttk.Label(f, textvariable=status, foreground="#333").grid(row=7, column=0, columnspan=5, sticky="w")

    def go():
        try:
            opts = dict(words=int(v_words.get()), links=int(v_links.get()), skip_top=int(v_skip.get()),
                        punct=v_punct.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Word map", "The settings must be whole numbers.")
            return
        path = model.get().strip()
        btn.configure(state="disabled")
        bar.start(12)

        def work():
            try:
                out, n, e = make_map(path, progress=lambda m: root.after(0, status.set, m), **opts)
                root.after(0, done, out, None)
            except Exception as ex:
                root.after(0, done, None, ex)
        threading.Thread(target=work, daemon=True).start()

    def done(out, err):
        bar.stop()
        bar.configure(value=0)
        btn.configure(state="normal")
        if err:
            status.set(f"Problem: {err}")
            messagebox.showerror("Word map", str(err))
            return
        last["path"] = out
        open_btn.configure(state="normal")
        webbrowser.open(out.as_uri())

    btn.configure(command=go)
    root.mainloop()


def main():
    if len(sys.argv) == 1:                 # double-clicked: show the window
        window()
        return
    ap = argparse.ArgumentParser(description="3D word-association map from an ngram.py model.")
    ap.add_argument("model", help="model.db made by ngram.py build")
    ap.add_argument("-o", "--output", help="default: <model>_map.html next to the model")
    ap.add_argument("--words", type=int, default=500, help="how many words to place (default 500)")
    ap.add_argument("--links", type=int, default=5, help="strongest links kept per word (default 5)")
    ap.add_argument("--punct", action="store_true", help="include punctuation marks as words")
    ap.add_argument("--skip-top", type=int, default=0,
                    help="leave out the N most common words (e.g. 20 hides 'the', 'of', 'and'...)")
    ap.add_argument("--window", action="store_true", help="open the window with this model selected")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()
    if args.window:
        window(args.model)
        return
    try:
        out, n, e = make_map(args.model, args.output, args.words, args.links, args.punct, args.skip_top)
    except (FileNotFoundError, ValueError) as ex:
        sys.exit(str(ex))
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
