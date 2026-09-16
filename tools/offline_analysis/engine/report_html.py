"""A self-contained interactive report for an analysis batch.

One ``.html`` file that opens in any browser with **no dependencies at all**,
no matplotlib, no plotly, no CDN, no network. The data is embedded as JSON and
drawn with inline SVG and vanilla JS.

That is not a stylistic preference. matplotlib is absent from some of the
environments this analysis runs in, and when it is absent the PNG path
produces nothing and (before this) said nothing. A report that needs no
plotting library is the only one that works everywhere the analysis does.

**It computes nothing.** Every number here comes from
:mod:`source.analysis.offline_analysis`, the summary rows, the bins, the
track arrays and the bout detectors. This module reshapes and draws; if you
find arithmetic in here that is a bug, because it means a measure now has two
definitions that can disagree.

What interactivity buys over a PNG:

* zoom and pan a 40-minute trace instead of squinting at 600 px;
* brush a time window and see the measures for **that window**;
* hover a bout to read its start and duration;
* sort/filter the batch table, click a row to load that animal;
* overlay two animals on one axis.
"""

from __future__ import annotations

import html
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["write_report", "build_report_html", "series_for"]

# Downsample long traces before embedding. A 40-minute session at 30 fps is
# 72k points; an SVG polyline with that many vertices is slow to draw and
# indistinguishable from one with 4k. The brush re-reads the full arrays, so
# this only bounds what is DRAWN, never what is measured.
MAX_POINTS = 4000


def _thin(n: int, cap: int = MAX_POINTS) -> np.ndarray:
    """Indices of an evenly-spaced subsample of ``n`` samples."""
    if n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).astype(int)


def _clean(a) -> List[Optional[float]]:
    """NaN/Inf → None, so the JSON is valid and gaps stay gaps.

    ``json.dumps`` emits bare ``NaN`` which no browser will parse, and
    replacing gaps with 0 would draw a line through missing tracking, the
    dropout would look like a visit to the origin.
    """
    out: List[Optional[float]] = []
    for v in np.asarray(a, dtype=float).ravel():
        out.append(None if not math.isfinite(v) else round(float(v), 4))
    return out


def series_for(sess, tr, params: dict) -> dict:
    """The per-session arrays the report draws, from an analysed track.

    Reads what ``build_track`` already produced.
    """
    from tools.offline_analysis.engine import offline_analysis as oa

    n = len(tr.cx)
    idx = _thin(n)
    t = np.cumsum(np.concatenate([[0.0], tr.dt[:-1]])) if n else np.zeros(0)

    out: Dict[str, Any] = {
        "stem": getattr(sess, "stem", ""),
        "subject": getattr(sess, "subject", "") or getattr(sess, "stem", ""),
        "n": int(n),
        "duration_s": float(tr.duration_s),
        "t": _clean(t[idx]),
        "x": _clean(tr.cx[idx]),
        "y": _clean(tr.cy[idx]),
        "speed": _clean(tr.speed_cm_s[idx]),
        "units": {"xy": "cm" if getattr(tr, "in_cm", True) else "px",
                  "speed": "cm/s"},
    }

    # Bouts, from the SAME detector the workbook used. Recomputing them with
    # different thresholds here is exactly how a figure comes to disagree with
    # the table it sits next to.
    try:
        _t, _n, _lat, spans = oa.freeze_bouts(
            tr.speed_cm_s, tr.dt, float(params.get("freeze_cm_s", 0.5)),
            float(params.get("min_freeze_s", 1.0)), return_spans=True)
        out["freeze_spans"] = spans
    except TypeError:
        out["freeze_spans"] = []          # older signature, no spans available
    except Exception:
        out["freeze_spans"] = []

    # Zone occupancy as spans, one row per zone.
    zones: Dict[str, List[List[float]]] = {}
    try:
        for z in oa.zone_names(sess):
            mask = oa._zone_mask(sess, z, n)
            zones[z] = _spans(np.asarray(mask, bool), t)
    except Exception:
        pass
    out["zones"] = zones
    return out


def _spans(mask: np.ndarray, t: np.ndarray) -> List[List[float]]:
    """Contiguous True runs of ``mask`` as ``[[t_start, t_end], …]``."""
    if mask.size == 0 or t.size == 0:
        return []
    out: List[List[float]] = []
    start: Optional[float] = None
    for i, v in enumerate(mask):
        ti = float(t[min(i, len(t) - 1)])
        if v and start is None:
            start = ti
        elif not v and start is not None:
            out.append([round(start, 3), round(ti, 3)])
            start = None
    if start is not None:
        out.append([round(start, 3), round(float(t[-1]), 3)])
    return out


# ── page ────────────────────────────────────────────────────────────────

def build_report_html(payload: dict, title: str = "Analysis report") -> str:
    """The whole page as one string. ``payload`` is embedded verbatim."""
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    # </script> inside the data would end the tag early.
    blob = blob.replace("<", "\\u003c").replace(">", "\\u003e")
    return _TEMPLATE.replace("__TITLE__", html.escape(title)) \
                    .replace("__DATA__", blob)


def write_report(res: dict, out_path: str, params: Optional[dict] = None,
                 sessions: Optional[List[dict]] = None) -> str:
    """Write the report for an ``analyze_files`` result.

    ``sessions`` carries the per-session traces when the caller has them
    (``preview_one``/``analyze3d``). Without them the report still renders the
    batch table, the bins and the input record, degraded, never broken, because
    a report that refuses to open is worse than one with fewer panels.
    """
    # EVERYTHING goes through _jsonable, not just the traces. A summary row
    # legitimately carries NaN (a latency for a bout that never happened), and
    # json.dumps emits a bare NaN that no browser will parse, so one absent
    # measure would produce a zero-byte report with no error visible to the
    # user. Learned the hard way: it did.
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "summary": _jsonable(res.get("summary") or []),
        "bins": _jsonable(res.get("bins") or []),
        "excluded": _jsonable(res.get("excluded") or []),
        "inputs": _jsonable(res.get("inputs") or []),
        "plot_state": res.get("plot_state", ""),
        "plot_detail": res.get("plot_detail", ""),
        "params": _jsonable(params if params is not None
                            else res.get("params") or {}),
        "sessions": _jsonable(sessions or []),
    }
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Render BEFORE opening the file: a failure half-way through must not
    # leave a truncated or empty .html sitting there looking like a result.
    page = build_report_html(payload)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out_path


def _jsonable(obj):
    """numpy scalars and NaN are not JSON; make them so without losing gaps."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f8fa;--panel:#fff;--ink:#15181f;--muted:#59616e;--rule:#e3e7ee;
  --accent:#0f766e;--accent-weak:#e6f4f1;--warn:#b42318;--warn-weak:#fdecea;
  --shadow:0 1px 2px rgba(16,20,30,.05),0 10px 28px rgba(16,20,30,.06)}
@media(prefers-color-scheme:dark){:root{--bg:#0c0f14;--panel:#141922;--ink:#eaeef5;
  --muted:#98a2b3;--rule:#232a35;--accent:#5eead4;--accent-weak:#0d2b28;
  --warn:#ff8a80;--warn-weak:#2a1414;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 14px 40px rgba(0,0,0,.5)}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.6 "Segoe UI",-apple-system,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 90px}
h1{font-size:30px;margin:6px 0 4px;letter-spacing:-.02em}
h2{font-size:20px;margin:34px 0 8px;padding-top:14px;border-top:1px solid var(--rule)}
.sub{color:var(--muted);margin:0 0 6px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:12px;
 padding:14px 16px;box-shadow:var(--shadow);margin:14px 0}
.note{background:var(--warn-weak);border-left:4px solid var(--warn);
 border-radius:8px;padding:10px 14px;margin:14px 0}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
select,input,button{font:inherit;padding:5px 9px;border-radius:7px;
 border:1px solid var(--rule);background:var(--panel);color:var(--ink)}
button{cursor:pointer}
button.on{background:var(--accent-weak);border-color:var(--accent);color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13.4px}
th,td{padding:7px 9px;border-bottom:1px solid var(--rule);text-align:left;
 white-space:nowrap}
th{font:700 10.5px/1 ui-monospace,Consolas,monospace;letter-spacing:.06em;
 text-transform:uppercase;color:var(--muted);cursor:pointer;user-select:none}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--accent-weak)}
tbody tr.sel{background:var(--accent-weak);box-shadow:inset 3px 0 0 var(--accent)}
.scroll{overflow:auto;max-height:420px;border:1px solid var(--rule);border-radius:11px}
svg{display:block;width:100%;background:var(--panel);border:1px solid var(--rule);
 border-radius:11px}
.k{font:11px ui-monospace,Consolas,monospace;fill:var(--muted)}
.tip{position:fixed;pointer-events:none;background:var(--panel);color:var(--ink);
 border:1px solid var(--rule);border-radius:7px;padding:5px 8px;font-size:12.5px;
 box-shadow:var(--shadow);opacity:0;transition:opacity .1s;z-index:9}
.stat{display:inline-block;margin:0 22px 8px 0}
.stat b{display:block;font-size:21px;letter-spacing:-.01em}
.stat span{font-size:11.5px;color:var(--muted)}
.muted{color:var(--muted)}
details{margin:10px 0}
summary{cursor:pointer;color:var(--accent);font-size:13.5px}
pre{overflow:auto;font:12px ui-monospace,Consolas,monospace;background:var(--panel);
 border:1px solid var(--rule);border-radius:9px;padding:12px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub" id="sub"></p>
<div id="warn"></div>

<div class="card" id="statbar"></div>

<h2>Batch</h2>
<p class="sub">Click a row to load that session. Click a header to sort.</p>
<div class="row" style="margin-bottom:8px">
  <input id="filter" placeholder="filter rows…" style="flex:1 1 240px">
  <button id="csv">Copy as TSV</button>
</div>
<div class="scroll"><table id="tbl"><thead></thead><tbody></tbody></table></div>

<h2>Session</h2>
<div class="row" id="ctrls">
  <select id="pick"></select>
  <button id="bZone" class="on">Zones</button>
  <button id="bFreeze" class="on">Freezing</button>
  <select id="overlay"><option value="">overlay: none</option></select>
  <button id="reset">Reset zoom</button>
</div>
<p class="sub" id="brushinfo">Drag on a trace to select a time window, the
  window measures update below. Double-click to clear.</p>
<div class="card" id="winstats"></div>
<svg id="sSpeed" height="150" role="img" aria-label="speed over time"></svg>
<svg id="sTrack" height="360" role="img" aria-label="track"></svg>

<h2>Inputs</h2>
<div class="card" id="prov"></div>
<div class="tip" id="tip"></div>
</div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const el = (n, a = {}) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const fmt = v => v === null || v === undefined || Number.isNaN(v) ? ", "
  : (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0)
     : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3)) : v);

/* ---------- header ---------- */
$("#sub").textContent =
  `${D.summary.length} session(s) · generated ${D.generated}`;
if (D.plot_state && D.plot_state !== "ok") {
  $("#warn").innerHTML = `<div class="note"><b>No PNG figures were written.</b> `
    + `${D.plot_detail || D.plot_state}. This report does not need them.</div>`;
}

/* ---------- batch table ---------- */
const rows = D.summary.slice();
let cols = rows.length ? Object.keys(rows[0]) : [];
let sortKey = cols[0], sortDir = 1;
function drawTable() {
  const q = $("#filter").value.toLowerCase();
  const body = rows
    .filter(r => !q || Object.values(r).some(v => String(v).toLowerCase().includes(q)))
    .sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      if (typeof x === "number" && typeof y === "number") return (x - y) * sortDir;
      return String(x).localeCompare(String(y)) * sortDir;
    });
  $("#tbl thead").innerHTML = "<tr>" + cols.map(c =>
    `<th data-k="${c}">${c}${c === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`
  ).join("") + "</tr>";
  $("#tbl tbody").innerHTML = body.map(r =>
    `<tr data-stem="${r.File || r.Subject || ""}">`
    + cols.map(c => `<td>${fmt(r[c])}</td>`).join("") + "</tr>").join("");
  $("#tbl thead").querySelectorAll("th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    sortDir = (k === sortKey) ? -sortDir : 1; sortKey = k; drawTable();
  });
  $("#tbl tbody").querySelectorAll("tr").forEach(tr => tr.onclick = () => {
    const i = D.sessions.findIndex(s =>
      s.stem === tr.dataset.stem || s.subject === tr.dataset.stem);
    if (i >= 0) { $("#pick").value = String(i); loadSession(); }
    $("#tbl tbody").querySelectorAll("tr").forEach(o => o.classList.remove("sel"));
    tr.classList.add("sel");
  });
}
$("#filter").oninput = drawTable;
$("#csv").onclick = () => {
  const txt = [cols.join("\t")].concat(
    rows.map(r => cols.map(c => r[c]).join("\t"))).join("\n");
  navigator.clipboard?.writeText(txt);
  $("#csv").textContent = "Copied"; setTimeout(() => $("#csv").textContent = "Copy as TSV", 1200);
};
drawTable();

/* ---------- stat bar ---------- */
function statbar(list) {
  const num = k => list.map(r => r[k]).filter(v => typeof v === "number");
  const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
  const items = [
    ["Sessions", list.length],
    ["Mean distance (m)", mean(num("Distance_m"))],
    ["Mean speed (cm/s)", mean(num("Mean_Speed_cm_s"))],
    ["Mean duration (s)", mean(num("Duration_s"))],
  ];
  $("#statbar").innerHTML = items.map(([l, v]) =>
    `<div class="stat"><b>${fmt(v)}</b><span>${l}</span></div>`).join("");
}
statbar(rows);

/* ---------- session picker ---------- */
const S = D.sessions || [];
$("#pick").innerHTML = S.length
  ? S.map((s, i) => `<option value="${i}">${s.subject || s.stem}</option>`).join("")
  : `<option>no traces embedded</option>`;
$("#overlay").innerHTML = `<option value="">overlay: none</option>` +
  S.map((s, i) => `<option value="${i}">overlay: ${s.subject || s.stem}</option>`).join("");

let cur = S[0] || null, ov = null, brush = null;
const show = { zone: true, freeze: true };
for (const [id, k] of [["#bZone", "zone"], ["#bFreeze", "freeze"]])
  $(id).onclick = () => { show[k] = !show[k]; $(id).classList.toggle("on", show[k]); draw(); };
$("#pick").onchange = loadSession;
$("#overlay").onchange = () => { ov = $("#overlay").value === "" ? null : S[+$("#overlay").value]; draw(); };
$("#reset").onclick = () => { brush = null; draw(); };
function loadSession() { cur = S[+$("#pick").value] || null; brush = null; draw(); }

/* ---------- drawing ---------- */
const tip = $("#tip");
function hover(node, text) {
  node.onmousemove = e => { tip.textContent = text; tip.style.opacity = 1;
    tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px"; };
  node.onmouseleave = () => tip.style.opacity = 0;
}
function axes(svg, W, H, pad, x0, x1, y0, y1, xlab, ylab) {
  svg.appendChild(el("line", {x1: pad, y1: H - pad, x2: W - 6, y2: H - pad,
    stroke: "currentColor", "stroke-opacity": .25}));
  svg.appendChild(el("line", {x1: pad, y1: 6, x2: pad, y2: H - pad,
    stroke: "currentColor", "stroke-opacity": .25}));
  for (const [v, lab] of [[y0, fmt(y0)], [y1, fmt(y1)]]) {
    const yy = H - pad - (v - y0) / ((y1 - y0) || 1) * (H - pad - 10);
    const t = el("text", {x: 4, y: yy + 4, class: "k"}); t.textContent = lab;
    svg.appendChild(t);
  }
  const tx = el("text", {x: W - 8, y: H - 6, class: "k", "text-anchor": "end"});
  tx.textContent = xlab; svg.appendChild(tx);
  const ty = el("text", {x: pad + 4, y: 14, class: "k"}); ty.textContent = ylab;
  svg.appendChild(ty);
}
function line(svg, xs, ys, W, H, pad, x0, x1, y0, y1, color, width) {
  let d = "", pen = false;
  for (let i = 0; i < xs.length; i++) {
    const yv = ys[i];
    if (yv === null) { pen = false; continue; }
    const X = pad + (xs[i] - x0) / ((x1 - x0) || 1) * (W - pad - 8);
    const Y = H - pad - (yv - y0) / ((y1 - y0) || 1) * (H - pad - 10);
    d += (pen ? "L" : "M") + X.toFixed(1) + " " + Y.toFixed(1) + " ";
    pen = true;
  }
  svg.appendChild(el("path", {d, fill: "none", stroke: color,
    "stroke-width": width || 1.4, "stroke-linejoin": "round"}));
}
function spanRects(svg, spans, W, H, pad, x0, x1, color, label) {
  for (const [a, b] of spans || []) {
    if (b < x0 || a > x1) continue;
    const X = pad + (Math.max(a, x0) - x0) / ((x1 - x0) || 1) * (W - pad - 8);
    const X2 = pad + (Math.min(b, x1) - x0) / ((x1 - x0) || 1) * (W - pad - 8);
    const r = el("rect", {x: X, y: 8, width: Math.max(1, X2 - X), height: H - pad - 8,
      fill: color, "fill-opacity": .18});
    hover(r, `${label} ${a.toFixed(1)}–${b.toFixed(1)} s (${(b - a).toFixed(2)} s)`);
    svg.appendChild(r);
  }
}
function attachBrush(svg, W, pad, x0, x1) {
  let down = null;
  svg.onmousedown = e => { down = e.offsetX; };
  svg.ondblclick = () => { brush = null; draw(); };
  svg.onmouseup = e => {
    if (down === null) return;
    const a = Math.min(down, e.offsetX), b = Math.max(down, e.offsetX);
    down = null;
    if (b - a < 4) return;
    const inv = px => x0 + (px - pad) / ((W - pad - 8) || 1) * (x1 - x0);
    brush = [inv(a), inv(b)]; draw();
  };
}
function windowStats(s) {
  if (!s) return;
  const [a, b] = brush || [0, s.duration_s];
  let n = 0, sum = 0, mx = 0;
  for (let i = 0; i < s.t.length; i++) {
    const t = s.t[i]; if (t === null || t < a || t > b) continue;
    const v = s.speed[i];
    if (v !== null) { n++; sum += v; mx = Math.max(mx, v); }
  }
  const items = [["Window (s)", (b - a).toFixed(1)],
                 ["Mean speed (cm/s)", n ? sum / n : null],
                 ["Max speed (cm/s)", n ? mx : null]];
  $("#winstats").innerHTML =
    `<div class="muted" style="font-size:12.5px;margin-bottom:6px">`
    + (brush ? `selected ${a.toFixed(1)}–${b.toFixed(1)} s` : "whole session")
    + `</div>` + items.map(([l, v]) =>
        `<div class="stat"><b>${fmt(v)}</b><span>${l}</span></div>`).join("");
}
function draw() {
  for (const id of ["#sSpeed", "#sTrack"]) $(id).innerHTML = "";
  const s = cur;
  if (!s) { $("#winstats").innerHTML = `<span class="muted">No traces embedded, `
    + `run the CLI or analyze3d.py to include them.</span>`; return; }
  const x0 = brush ? brush[0] : 0, x1 = brush ? brush[1] : (s.duration_s || 1);

  /* speed */
  {
    const svg = $("#sSpeed"), W = svg.clientWidth || 900, H = 150, pad = 42;
    const ys = s.speed.filter(v => v !== null);
    const y1 = Math.max(1, ...ys);
    axes(svg, W, H, pad, x0, x1, 0, y1, "time (s)", "speed cm/s");
    if (show.zone) {
      const names = Object.keys(s.zones || {});
      names.forEach((z, i) => spanRects(svg, s.zones[z], W, H, pad, x0, x1,
        `hsl(${(i * 67) % 360} 70% 55%)`, z));
    }
    if (show.freeze) spanRects(svg, s.freeze_spans, W, H, pad, x0, x1,
      "var(--warn)", "freeze");
    line(svg, s.t, s.speed, W, H, pad, x0, x1, 0, y1, "var(--accent)");
    if (ov) line(svg, ov.t, ov.speed, W, H, pad, x0, x1, 0, y1, "#8b7ef0", 1.1);
    attachBrush(svg, W, pad, x0, x1);
  }
  /* track */
  {
    const svg = $("#sTrack"), W = svg.clientWidth || 900, H = 360, pad = 42;
    const xs = [], ys = [], sp = [];
    for (let i = 0; i < s.t.length; i++) {
      const t = s.t[i]; if (t === null || t < x0 || t > x1) continue;
      if (s.x[i] === null || s.y[i] === null) continue;
      xs.push(s.x[i]); ys.push(s.y[i]); sp.push(s.speed[i] ?? 0);
    }
    if (xs.length > 1) {
      const ax0 = Math.min(...xs), ax1 = Math.max(...xs);
      const ay0 = Math.min(...ys), ay1 = Math.max(...ys);
      const sx = v => pad + (v - ax0) / ((ax1 - ax0) || 1) * (W - pad - 14);
      const sy = v => H - pad - (v - ay0) / ((ay1 - ay0) || 1) * (H - pad - 14);
      const mx = Math.max(1, ...sp);
      for (let i = 1; i < xs.length; i++) {
        const c = Math.round(240 - 240 * (sp[i] / mx));
        svg.appendChild(el("line", {x1: sx(xs[i - 1]), y1: sy(ys[i - 1]),
          x2: sx(xs[i]), y2: sy(ys[i]), stroke: `hsl(${c} 80% 50%)`,
          "stroke-width": 1.5, "stroke-linecap": "round"}));
      }
      const lab = el("text", {x: pad + 4, y: 16, class: "k"});
      lab.textContent = `track (${s.units.xy}), colour = speed`;
      svg.appendChild(lab);
    }
  }
  windowStats(s);
}
loadSession();
window.addEventListener("resize", draw);

/* ---------- input record ---------- */
$("#prov").innerHTML =
  `<div class="muted" style="font-size:12.5px">Inputs</div>`
  + (D.inputs.length
      ? `<table><thead><tr><th>file</th><th>bytes</th><th>modified</th><th>sha256</th></tr></thead><tbody>`
        + D.inputs.map(i => `<tr><td>${i.file}</td><td>${fmt(i.bytes)}</td>`
          + `<td>${i.mtime || ": "}</td><td>${i.sha256 || ": "}</td></tr>`).join("")
        + `</tbody></table>`
      : `<span class="muted">none recorded</span>`)
  + `<details><summary>Parameters that produced this</summary><pre>`
  + JSON.stringify(D.params, null, 2).replace(/</g, "&lt;") + `</pre></details>`;
</script></body></html>
"""
