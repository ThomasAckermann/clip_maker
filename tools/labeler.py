"""
Multi-clip video labeling tool for volleyball action events.

Modes:
    Single clip:  python tools/labeler.py --video path/to/rally.mp4 --output labels.json
    Multi-clip:   python tools/labeler.py --clips-dir path/to/clips/ --output labels.json

Opens http://localhost:8000 in your browser.

Keyboard shortcuts:
    Space           play / pause
    ← / →           step one frame back / forward
    Shift + ← / →   jump 1 second back / forward
    L               focus the label dropdown
    N               next clip
    P               previous clip
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

CLASSES = ["background", "block", "receive", "score", "serve", "set", "spike"]


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Browser-based video event labeler")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Single input video file.")
    group.add_argument("--clips-dir", type=Path, help="Directory of MP4 clips to label.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("labels.json"),
        help="Output JSON file (default: labels.json).",
    )
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


# ── Video metadata ────────────────────────────────────────────────────────────


def get_video_info(video_path: Path) -> tuple[float, float]:
    """Return (fps, duration_sec) via ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            avg_fr = stream.get("avg_frame_rate", "30/1")
            num, den = avg_fr.split("/")
            den_f = float(den)
            if den_f == 0:
                raise RuntimeError(f"ffprobe returned invalid frame rate: {avg_fr!r}")
            return float(num) / den_f, float(stream.get("duration", 0))
    raise RuntimeError("No video stream found.")


# ── State ─────────────────────────────────────────────────────────────────────


class _State:
    """Mutable server-side state for the active clip and clip list."""

    def __init__(self, clips: list[tuple[str, Path]], output_path: Path) -> None:
        self.clips = clips  # [(name, path), …]
        self.output_path = output_path
        self._idx = 0
        self._meta: dict[str, tuple[float, float, int]] = {}  # (fps, duration, num_frames)

    # ── Active clip ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.clips[self._idx][0]

    @property
    def path(self) -> Path:
        return self.clips[self._idx][1]

    @property
    def idx(self) -> int:
        return self._idx

    def switch_to(self, name: str) -> bool:
        for i, (n, _) in enumerate(self.clips):
            if n == name:
                self._idx = i
                return True
        return False

    # ── Metadata (lazy, cached) ───────────────────────────────────────────────

    def meta(self, name: str | None = None) -> tuple[float, float, int]:
        n = name or self.name
        if n not in self._meta:
            path = {k: v for k, v in self.clips}[n]
            fps, dur = get_video_info(path)
            self._meta[n] = (fps, dur, int(fps * dur))
        return self._meta[n]

    # ── Persisted events ──────────────────────────────────────────────────────

    def load_events(self, name: str | None = None) -> list[dict]:
        """Return saved events for a clip as frontend-ready dicts {frame,label,x,y}."""
        n = name or self.name
        if not self.output_path.exists():
            return []
        try:
            for entry in json.loads(self.output_path.read_text()):
                if entry.get("video") == n:
                    return [
                        {"frame": e["frame"], "label": e["label"], "x": e["xy"][0], "y": e["xy"][1]}
                        for e in entry.get("events", [])
                    ]
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return []

    def is_labeled(self, name: str) -> bool:
        return bool(self.load_events(name))

    def save_events(self, name: str, events: list[dict]) -> None:
        fps, _, num_frames = self.meta(name)
        entry = {
            "video": name,
            "num_frames": num_frames,
            "fps": fps,
            "events": [
                {
                    "frame": e["frame"],
                    "label": e["label"],
                    "xy": [round(e["x"], 4), round(e["y"], 4)],
                }
                for e in events
            ],
        }
        existing: list = []
        if self.output_path.exists():
            try:
                existing = [
                    e for e in json.loads(self.output_path.read_text()) if e.get("video") != name
                ]
            except (json.JSONDecodeError, ValueError):
                pass
        existing.append(entry)
        self.output_path.write_text(json.dumps(existing, indent=2))


# ── FastAPI app ───────────────────────────────────────────────────────────────


def make_app(clips: list[tuple[str, Path]], output_path: Path) -> FastAPI:
    state = _State(clips, output_path)
    # Pre-load metadata for the first clip so the first page load is instant
    state.meta()

    app = FastAPI()

    # ── Clips list ────────────────────────────────────────────────────────────

    @app.get("/clips")
    def list_clips():
        result = []
        for name, _ in state.clips:
            fps, dur, _ = state.meta(name)
            result.append(
                {
                    "name": name,
                    "labeled": state.is_labeled(name),
                    "duration": round(dur, 1),
                    "active": name == state.name,
                }
            )
        return result

    @app.post("/switch/{name}")
    def switch_clip(name: str):
        if not state.switch_to(name):
            return JSONResponse({"error": f"Unknown clip: {name}"}, status_code=404)
        fps, duration, num_frames = state.meta()
        return {
            "name": state.name,
            "fps": fps,
            "duration": duration,
            "num_frames": num_frames,
            "idx": state.idx,
            "total": len(state.clips),
            "events": state.load_events(),
        }

    # ── Current clip info + events ────────────────────────────────────────────

    @app.get("/info")
    def info():
        fps, duration, num_frames = state.meta()
        return {
            "name": state.name,
            "fps": fps,
            "duration": duration,
            "num_frames": num_frames,
            "idx": state.idx,
            "total": len(state.clips),
            "classes": CLASSES,
            "events": state.load_events(),
        }

    # ── Video streaming (range request support required for seeking) ──────────

    @app.get("/video")
    async def stream_video(request: Request):
        video_path = state.path
        file_size = video_path.stat().st_size
        range_hdr = request.headers.get("Range")

        if range_hdr:
            m = re.search(r"(\d+)-(\d*)", range_hdr)
            byte1 = int(m.group(1))
            byte2 = int(m.group(2)) if m.group(2) else file_size - 1
            length = byte2 - byte1 + 1

            def _stream():
                with open(video_path, "rb") as f:
                    f.seek(byte1)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                _stream(),
                status_code=206,
                media_type="video/mp4",
                headers={
                    "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                },
            )

        return StreamingResponse(
            open(video_path, "rb"),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    # ── Save ──────────────────────────────────────────────────────────────────

    @app.post("/save")
    async def save(request: Request):
        body = await request.json()
        name = body.get("name", state.name)
        events = body.get("events", [])
        state.save_events(name, events)
        return {"saved": str(output_path), "num_events": len(events), "name": name}

    @app.get("/", response_class=HTMLResponse)
    def index():
        multi = len(clips) > 1
        return _build_page(multi)

    return app


# ── HTML/JS UI ────────────────────────────────────────────────────────────────


def _build_page(multi_clip: bool) -> str:
    clips_panel_display = "flex" if multi_clip else "none"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Volleyball Event Labeler</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #1a1a1a; color: #e0e0e0;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}

  #header {{ padding: 8px 14px; background: #111; border-bottom: 1px solid #333;
            display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
  #header h1 {{ font-size: 0.95rem; font-weight: 600; color: #fff; }}
  #clip-title {{ font-size: 0.85rem; color: #4a9eff; font-weight: 600; }}
  #clip-counter {{ font-size: 0.75rem; color: #555; margin-left: auto; }}

  #main {{ display: flex; flex: 1; overflow: hidden; }}

  /* ── Clips sidebar ── */
  #clips-panel {{ width: 200px; display: {clips_panel_display}; flex-direction: column;
                  background: #181818; border-right: 1px solid #333; overflow: hidden; }}
  #clips-header {{ padding: 8px 10px; font-size: 0.7rem; text-transform: uppercase;
                   color: #666; letter-spacing: 0.05em; border-bottom: 1px solid #2a2a2a; flex-shrink: 0; }}
  #clips-list {{ flex: 1; overflow-y: auto; padding: 4px; }}
  .clip-item {{ display: flex; align-items: center; gap: 6px; padding: 6px 8px;
                border-radius: 5px; cursor: pointer; font-size: 0.78rem;
                margin-bottom: 2px; }}
  .clip-item:hover {{ background: #2a2a2a; }}
  .clip-item.active {{ background: #1e3a5f; color: #fff; }}
  .clip-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
               background: #444; }}
  .clip-dot.labeled {{ background: #2ecc71; }}
  .clip-name {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .clip-dur {{ color: #555; font-size: 0.7rem; flex-shrink: 0; }}
  #nav-row {{ display: flex; gap: 4px; padding: 6px; border-top: 1px solid #2a2a2a; flex-shrink: 0; }}
  .btn-nav {{ flex: 1; padding: 6px; background: #2a2a2a; border: 1px solid #444;
              border-radius: 5px; color: #aaa; cursor: pointer; font-size: 0.78rem; }}
  .btn-nav:hover {{ background: #333; color: #fff; }}
  .btn-nav:disabled {{ opacity: 0.3; cursor: default; }}

  /* ── Video panel ── */
  #video-panel {{ flex: 1; display: flex; flex-direction: column;
                 background: #000; position: relative; min-width: 0; }}
  #video-wrap {{ position: relative; flex: 1; display: flex;
                align-items: center; justify-content: center; overflow: hidden; }}
  #vid {{ display: block; max-width: 100%; max-height: 100%; cursor: crosshair; }}
  #overlay {{ position: absolute; cursor: crosshair; pointer-events: none; }}
  #timeline {{ padding: 8px 12px; background: #111; flex-shrink: 0; }}
  input[type=range] {{ width: 100%; accent-color: #4a9eff; }}
  #time-info {{ display: flex; justify-content: space-between;
               font-size: 0.72rem; color: #666; margin-top: 2px; }}

  /* ── Controls panel ── */
  #controls {{ width: 300px; display: flex; flex-direction: column;
              background: #222; border-left: 1px solid #333; overflow: hidden; flex-shrink: 0; }}

  #frame-info {{ padding: 10px 12px; background: #1a1a1a; border-bottom: 1px solid #333; flex-shrink: 0; }}
  #frame-info .row {{ display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 3px; }}
  #frame-info .row span:first-child {{ color: #666; }}
  #frame-num {{ font-size: 1.3rem; font-weight: 700; color: #4a9eff; }}

  #add-section {{ padding: 10px 12px; border-bottom: 1px solid #333; flex-shrink: 0; }}
  #add-section h3 {{ font-size: 0.7rem; text-transform: uppercase; color: #666;
                     margin-bottom: 7px; letter-spacing: 0.05em; }}
  select, button {{ width: 100%; padding: 7px 10px; border-radius: 5px;
                   font-size: 0.82rem; border: 1px solid #444; }}
  select {{ background: #2a2a2a; color: #e0e0e0; margin-bottom: 7px; }}
  #xy-display {{ font-size: 0.72rem; color: #888; margin-bottom: 7px; min-height: 1.1em; }}
  #btn-add {{ background: #4a9eff; color: #fff; border-color: #4a9eff;
             cursor: pointer; font-weight: 600; }}
  #btn-add:hover {{ background: #3a8eef; }}
  #btn-add:disabled {{ background: #2a2a2a; color: #555; border-color: #444; cursor: default; }}

  #events-section {{ flex: 1; overflow-y: auto; padding: 10px 12px; }}
  #events-section h3 {{ font-size: 0.7rem; text-transform: uppercase; color: #666;
                        margin-bottom: 7px; letter-spacing: 0.05em; }}
  .event-row {{ display: flex; align-items: center; gap: 5px; padding: 5px 7px;
               background: #2a2a2a; border-radius: 4px; margin-bottom: 3px;
               font-size: 0.76rem; cursor: pointer; }}
  .event-row:hover {{ background: #333; }}
  .event-frame {{ color: #4a9eff; font-weight: 700; width: 48px; flex-shrink: 0; }}
  .event-label {{ flex: 1; }}
  .event-xy {{ color: #666; width: 72px; flex-shrink: 0; }}
  .btn-del {{ background: none; border: none; color: #555; cursor: pointer;
             padding: 0 3px; font-size: 0.9rem; width: auto; }}
  .btn-del:hover {{ color: #ff5555; }}
  #no-events {{ color: #444; font-size: 0.78rem; text-align: center; margin-top: 14px; }}

  #save-section {{ padding: 10px 12px; border-top: 1px solid #333; flex-shrink: 0; }}
  #btn-save {{ background: #2ecc71; color: #fff; border-color: #2ecc71;
              cursor: pointer; font-weight: 600; }}
  #btn-save:hover {{ background: #27ae60; }}
  #save-msg {{ font-size: 0.72rem; color: #2ecc71; margin-top: 5px;
              min-height: 1.1em; text-align: center; }}

  #shortcuts {{ padding: 6px 12px; background: #111; border-top: 1px solid #333;
               font-size: 0.68rem; color: #444; flex-shrink: 0; }}
  #shortcuts span {{ margin-right: 10px; }}
  kbd {{ background: #242424; border: 1px solid #3a3a3a; border-radius: 3px;
        padding: 1px 4px; font-size: 0.68rem; }}
</style>
</head>
<body>

<div id="header">
  <h1>Labeler</h1>
  <span id="clip-title">Loading…</span>
  <span id="clip-counter"></span>
</div>

<div id="main">

  <div id="clips-panel">
    <div id="clips-header">Clips</div>
    <div id="clips-list"></div>
    <div id="nav-row">
      <button class="btn-nav" id="btn-prev" onclick="navigate(-1)">← Prev</button>
      <button class="btn-nav" id="btn-next" onclick="navigate(1)">Next →</button>
    </div>
  </div>

  <div id="video-panel">
    <div id="video-wrap">
      <video id="vid" preload="metadata"></video>
      <canvas id="overlay"></canvas>
    </div>
    <div id="timeline">
      <input type="range" id="scrubber" min="0" step="1" value="0">
      <div id="time-info">
        <span id="time-cur">0:00.00</span>
        <span id="time-dur">0:00.00</span>
      </div>
    </div>
  </div>

  <div id="controls">
    <div id="frame-info">
      <div class="row"><span>Frame</span><span id="frame-num">0</span></div>
      <div class="row"><span>Time</span><span id="time-str">0:00.00</span></div>
    </div>

    <div id="add-section">
      <h3>Add Event</h3>
      <select id="label-select"></select>
      <div id="xy-display">Click video to set ball position</div>
      <button id="btn-add" disabled>Add Event</button>
    </div>

    <div id="events-section">
      <h3>Events (<span id="event-count">0</span>)</h3>
      <div id="events-list"><div id="no-events">No events yet</div></div>
    </div>

    <div id="save-section">
      <button id="btn-save">Save</button>
      <div id="save-msg"></div>
    </div>
  </div>
</div>

<div id="shortcuts">
  <span><kbd>Space</kbd> play/pause</span>
  <span><kbd>←</kbd><kbd>→</kbd> frame step</span>
  <span><kbd>Shift+←</kbd><kbd>Shift+→</kbd> ±1s</span>
  <span><kbd>L</kbd> label</span>
  <span><kbd>N</kbd><kbd>P</kbd> next/prev clip</span>
</div>

<script>
const vid      = document.getElementById('vid');
const overlay  = document.getElementById('overlay');
const ctx      = overlay.getContext('2d');
const scrubber = document.getElementById('scrubber');

let fps = 30, numFrames = 0, currentName = '', currentIdx = 0, totalClips = 1;
let pendingX = null, pendingY = null;
let events = [];
let allClips = [];

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function init() {{
  const info = await fetch('/info').then(r => r.json());
  applyInfo(info);

  const sel = document.getElementById('label-select');
  info.classes.forEach(c => {{
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    if (c === 'spike') opt.selected = true;
    sel.appendChild(opt);
  }});

  await refreshClipsList();
  loadEvents(info.events);
}}
init();

function applyInfo(info) {{
  fps        = info.fps;
  numFrames  = info.num_frames;
  currentName = info.name;
  currentIdx  = info.idx;
  totalClips  = info.total;

  document.getElementById('clip-title').textContent  = info.name;
  document.getElementById('clip-counter').textContent =
    totalClips > 1 ? `${{info.idx + 1}} / ${{info.total}}` : '';
  document.getElementById('time-dur').textContent = fmtTime(info.duration);
  scrubber.max   = numFrames - 1;
  scrubber.value = 0;

  document.getElementById('btn-prev').disabled = (currentIdx === 0);
  document.getElementById('btn-next').disabled = (currentIdx === totalClips - 1);

  // Reload video — append a cache-buster so the browser doesn't reuse the old stream
  vid.src = '/video?t=' + Date.now();
  vid.load();
}}

// ── Clips sidebar ─────────────────────────────────────────────────────────────
async function refreshClipsList() {{
  allClips = await fetch('/clips').then(r => r.json());
  const list = document.getElementById('clips-list');
  list.innerHTML = allClips.map(c => `
    <div class="clip-item ${{c.name === currentName ? 'active' : ''}}"
         onclick="switchClip('${{c.name}}')">
      <span class="clip-dot ${{c.labeled ? 'labeled' : ''}}"></span>
      <span class="clip-name" title="${{c.name}}">${{c.name}}</span>
      <span class="clip-dur">${{c.duration}}s</span>
    </div>
  `).join('');
  // Scroll active item into view
  const active = list.querySelector('.clip-item.active');
  if (active) active.scrollIntoView({{ block: 'nearest' }});
}}

async function switchClip(name) {{
  if (name === currentName) return;
  await autoSave();
  const info = await fetch('/switch/' + encodeURIComponent(name), {{ method: 'POST' }})
    .then(r => r.json());
  resetPending();
  applyInfo(info);
  loadEvents(info.events);
  await refreshClipsList();
}}

function navigate(dir) {{
  const idx = currentIdx + dir;
  if (idx < 0 || idx >= totalClips) return;
  switchClip(allClips[idx].name);
}}

// ── Video sync ────────────────────────────────────────────────────────────────
vid.addEventListener('loadedmetadata', positionOverlay);
vid.addEventListener('timeupdate', onTimeUpdate);
vid.addEventListener('seeked', redrawOverlay);
window.addEventListener('resize', positionOverlay);
scrubber.addEventListener('input', () => {{ vid.currentTime = parseInt(scrubber.value) / fps; }});

function positionOverlay() {{
  const vr = vid.getBoundingClientRect();
  const pr = vid.parentElement.getBoundingClientRect();
  overlay.style.left = (vr.left - pr.left) + 'px';
  overlay.style.top  = (vr.top  - pr.top)  + 'px';
  overlay.width  = vr.width;
  overlay.height = vr.height;
  redrawOverlay();
}}

function onTimeUpdate() {{
  const frame = timeToFrame(vid.currentTime);
  scrubber.value = frame;
  document.getElementById('frame-num').textContent = frame;
  const t = fmtTime(vid.currentTime);
  document.getElementById('time-cur').textContent = t;
  document.getElementById('time-str').textContent = t;
  redrawOverlay();
}}

function timeToFrame(t)  {{ return Math.min(Math.round(t * fps), numFrames - 1); }}
function frameToTime(f)  {{ return f / fps; }}
function fmtTime(t) {{
  const m = Math.floor(t / 60), s = (t % 60).toFixed(2).padStart(5, '0');
  return `${{m}}:${{s}}`;
}}

// ── Click to set ball position ────────────────────────────────────────────────
vid.addEventListener('click', e => {{
  const r = vid.getBoundingClientRect();
  pendingX = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  pendingY = Math.max(0, Math.min(1, (e.clientY - r.top)  / r.height));
  document.getElementById('xy-display').textContent =
    `Ball at (${{pendingX.toFixed(3)}}, ${{pendingY.toFixed(3)}})`;
  document.getElementById('btn-add').disabled = false;
  redrawOverlay();
}});

function resetPending() {{
  pendingX = pendingY = null;
  document.getElementById('xy-display').textContent = 'Click video to set ball position';
  document.getElementById('btn-add').disabled = true;
}}

function redrawOverlay() {{
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const cf = timeToFrame(vid.currentTime);

  events.forEach(e => {{
    const ox = e.x * overlay.width, oy = e.y * overlay.height;
    const near = Math.abs(e.frame - cf) < 3;
    ctx.beginPath();
    ctx.arc(ox, oy, 8, 0, Math.PI * 2);
    ctx.fillStyle = near ? 'rgba(74,158,255,0.8)' : 'rgba(74,158,255,0.2)';
    ctx.fill();
    ctx.strokeStyle = '#4a9eff'; ctx.lineWidth = 2; ctx.stroke();
    if (near) {{
      ctx.fillStyle = '#fff'; ctx.font = '11px system-ui';
      ctx.fillText(e.label, ox + 10, oy - 4);
    }}
  }});

  if (pendingX !== null) {{
    ctx.beginPath();
    ctx.arc(pendingX * overlay.width, pendingY * overlay.height, 10, 0, Math.PI * 2);
    ctx.strokeStyle = '#ff9f00'; ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
  }}
}}

// ── Add / delete events ───────────────────────────────────────────────────────
document.getElementById('btn-add').addEventListener('click', () => {{
  if (pendingX === null) return;
  events.push({{ frame: timeToFrame(vid.currentTime),
                label: document.getElementById('label-select').value,
                x: pendingX, y: pendingY }});
  events.sort((a, b) => a.frame - b.frame);
  resetPending();
  renderEvents();
  redrawOverlay();
}});

function loadEvents(evts) {{
  events = (evts || []).map(e => ({{
    frame: e.frame, label: e.label, x: e.x ?? (e.xy?.[0] ?? 0), y: e.y ?? (e.xy?.[1] ?? 0)
  }}));
  renderEvents();
  redrawOverlay();
}}

function renderEvents() {{
  const list = document.getElementById('events-list');
  document.getElementById('event-count').textContent = events.length;
  if (!events.length) {{
    list.innerHTML = '<div id="no-events">No events yet</div>'; return;
  }}
  list.innerHTML = events.map((e, i) => `
    <div class="event-row" onclick="jumpTo(${{i}})">
      <span class="event-frame">#${{e.frame}}</span>
      <span class="event-label">${{e.label}}</span>
      <span class="event-xy">(${{e.x.toFixed(2)}}, ${{e.y.toFixed(2)}})</span>
      <button class="btn-del" onclick="event.stopPropagation();deleteEvent(${{i}})" title="Delete">✕</button>
    </div>`).join('');
}}

function jumpTo(i) {{ vid.currentTime = frameToTime(events[i].frame); }}
function deleteEvent(i) {{ events.splice(i, 1); renderEvents(); redrawOverlay(); }}

// ── Save ──────────────────────────────────────────────────────────────────────
async function doSave(silent = false) {{
  const d = await fetch('/save', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ name: currentName, events }}),
  }}).then(r => r.json());
  if (!silent) {{
    const msg = document.getElementById('save-msg');
    msg.textContent = `Saved ${{d.num_events}} events`;
    setTimeout(() => msg.textContent = '', 3000);
  }}
  return d;
}}

async function autoSave() {{
  if (events.length > 0) await doSave(true);
}}

document.getElementById('btn-save').addEventListener('click', async () => {{
  await doSave(false);
  await refreshClipsList();
}});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') {{
    e.preventDefault(); vid.paused ? vid.play() : vid.pause();
  }} else if (e.code === 'ArrowRight') {{
    e.preventDefault(); vid.pause();
    vid.currentTime = Math.min(vid.duration, vid.currentTime + (e.shiftKey ? 1 : 1/fps));
  }} else if (e.code === 'ArrowLeft') {{
    e.preventDefault(); vid.pause();
    vid.currentTime = Math.max(0, vid.currentTime - (e.shiftKey ? 1 : 1/fps));
  }} else if (e.code === 'KeyL') {{
    document.getElementById('label-select').focus();
  }} else if (e.code === 'KeyN') {{
    navigate(1);
  }} else if (e.code === 'KeyP') {{
    navigate(-1);
  }}
}});
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if args.video:
        if not args.video.exists():
            raise SystemExit(f"Video not found: {args.video}")
        clips = [(args.video.stem, args.video)]
    else:
        if not args.clips_dir.exists():
            raise SystemExit(f"Directory not found: {args.clips_dir}")
        clips = sorted(
            [(p.stem, p) for p in args.clips_dir.glob("*.mp4")],
            key=lambda x: x[0],
        )
        if not clips:
            raise SystemExit(f"No MP4 files found in {args.clips_dir}")

    print(f"Clips  : {len(clips)}")
    print(f"Output : {args.output}")
    print(f"Opening http://localhost:{args.port} …")
    webbrowser.open(f"http://localhost:{args.port}")

    app = make_app(clips, args.output)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
