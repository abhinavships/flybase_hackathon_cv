#!/usr/bin/env python3
"""
Patches server.py so the OpenStreetMap spatial-grounding view is oriented
correctly, and adds real class filters + a vehicle picker.

Idempotent and anchor-based, so it can be re-run after server.py is
regenerated. Run from the directory containing server.py:

    python3 fix_geo_and_selection.py [path/to/server.py]

WHAT IT FIXES
-------------
1. Orientation. X_s / Y_s are an IMAGE frame: +X right, +Y *down*. The old
   code sent +Y to north and +X to east, which mirrors the scene -- that is
   why the trace cross rendered reversed and floated off the carriageway.
   Seen from above a nadir frame is a true, un-mirrored plan view, so the
   correct map is a pure rotation that honours Y-down:

       +X points along compass bearing H   ->   +Y points along H + 90

   H = 106.65 deg and the anchor below were fitted against the OSM
   centrelines of Gopal Hari Deshmukh Marg (primary, local bearing 113.7)
   and the Baner - Mahalunge tertiary arm (local bearing 23.0). Mean
   trajectory-to-carriageway residual: 5.8 m.

2. Class filters. class_final emits LGV / HGV-rigid / three-wheeler /
   pedestrian, none of which matched the hard-coded checkbox values, so
   those tracks were dropped from the map entirely. Categories are now
   derived from the data via CLASS_CANON.

3. Vehicle selection. Searchable track list, per-track detail, "show
   selected vehicle only" isolation, and fit-to-track on the map.
"""
import re
import sys
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "server.py")
src = PATH.read_text()
orig = src

if "JUNCTION_LAT" in src and "projectPoint" in src:
    print("[=] Already patched — nothing to do.")
    sys.exit(0)

applied, skipped = [], []


def sub(name, pattern, repl, count=1, flags=re.S):
    """Regex replace; records whether the anchor was found."""
    global src
    new, n = re.subn(pattern, repl, src, count=count, flags=flags)
    if n:
        src = new
        applied.append(name)
    else:
        skipped.append(name)
    return n


# --------------------------------------------------------------------------
# 1. Constants: anchor, frame heading, class canonicalisation
# --------------------------------------------------------------------------
CONSTS = r'''# --- Ground-truth anchor, fitted against OpenStreetMap centrelines ----------
# Junction of Gopal Hari Deshmukh Marg (primary, local bearing 113.7 deg) and
# the Baner - Mahalunge tertiary arm (local bearing 23.0 deg).
# FRAME_X0 / FRAME_Y0 is that same junction in the video's metric frame.
JUNCTION_LAT = 18.566057
JUNCTION_LON = 73.771440
FRAME_X0 = 9.99
FRAME_Y0 = 68.32
FRAME_HEADING_DEG = 106.65  # compass bearing of the video frame's +X axis

M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(JUNCTION_LAT))

# class_final from the tracker is messy (LGV, HGV-rigid, three-wheeler, ...).
# Fold it into the categories the UI filters on so nothing is silently dropped.
CLASS_CANON = {
    "car": "car", "sedan": "car", "hatchback": "car", "suv": "car",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "bike": "motorcycle",
    "scooter": "motorcycle", "two-wheeler": "motorcycle",
    "three-wheeler": "auto", "threewheeler": "auto", "auto": "auto",
    "autorickshaw": "auto", "auto-rickshaw": "auto", "rickshaw": "auto",
    "bus": "bus", "minibus": "bus",
    "lgv": "lgv", "lcv": "lgv", "van": "lgv", "pickup": "lgv",
    "truck": "hgv", "hgv": "hgv", "hgv-rigid": "hgv", "hgv-artic": "hgv",
    "hgv-articulated": "hgv", "lorry": "hgv", "trailer": "hgv",
    "cyclist": "cyclist", "bicycle": "cyclist", "cycle": "cyclist",
    "pedestrian": "pedestrian", "person": "pedestrian",
}

CLASS_LABELS = {
    "car": "Car", "motorcycle": "Motorcycle", "auto": "Auto / 3-wheeler",
    "bus": "Bus", "lgv": "LGV / Van", "hgv": "Truck / HGV",
    "cyclist": "Cyclist", "pedestrian": "Pedestrian", "other": "Other",
}


def canon_class(raw) -> str:
    return CLASS_CANON.get(str(raw).strip().lower(), "other")


METADATA = {'''

sub("constants", r"METADATA = \{", CONSTS.replace("\\", "\\\\"), count=1, flags=0)

sub("metadata.center",
    r'"center_lat": [\d.]+,\s*\n\s*"center_lon": [\d.]+,',
    '"center_lat": JUNCTION_LAT,\n    "center_lon": JUNCTION_LON,\n    "categories": [],',
    flags=0)

sub("metadata.bearing",
    r'"bearing_deg": [\d.]+,',
    '"frame_x0": FRAME_X0,\n    "frame_y0": FRAME_Y0,\n    "heading_deg": FRAME_HEADING_DEG,',
    flags=0)

# --------------------------------------------------------------------------
# 2. The georeferencing itself
# --------------------------------------------------------------------------
GEO = '''    # ------------------------------------------------------------------
    # Georeferencing: image-plane metric frame -> WGS84
    #
    # X_s / Y_s come out of the nadir video, so they are an IMAGE frame:
    # +X is to the right of the frame and +Y points DOWN the frame. Seen from
    # above that is still a true, un-mirrored plan view, so the frame maps to
    # the ground by a pure rotation -- provided the Y-down convention is
    # honoured: if +X points along compass bearing H, then +Y points along
    # H + 90 deg.
    #
    # The old code sent +Y straight to north and +X to east, which mirrors
    # the scene. That is why the trace cross came out reversed and off-road.
    # ------------------------------------------------------------------
    x_c = df["X_s"] - FRAME_X0
    y_c = df["Y_s"] - FRAME_Y0

    h1 = math.radians(FRAME_HEADING_DEG)
    h2 = math.radians(FRAME_HEADING_DEG + 90.0)
    east = x_c * math.sin(h1) + y_c * math.sin(h2)
    north = x_c * math.cos(h1) + y_c * math.cos(h2)

    df["lat"] = JUNCTION_LAT + north / M_PER_DEG_LAT
    df["lon"] = JUNCTION_LON + east / M_PER_DEG_LON

    df["cat"] = df["class_final"].map(canon_class)
'''

sub("geo transform",
    r"[ \t]*#[^\n]*Calibration Center[^\n]*\n.*?df\[\"lon\"\] = LON_0 \+ \([^\n]*\n",
    GEO.replace("\\", "\\\\"))

# --------------------------------------------------------------------------
# 3. Category metadata + per-record / per-track category fields
# --------------------------------------------------------------------------
sub("metadata.categories",
    r'(METADATA\["classes"\] = sorted\(\[str\(c\) for c in df\["class_final"\]\.unique\(\)\]\))',
    r'''\1
    _cat_counts = df.groupby("cat")["track_id"].nunique()
    METADATA["categories"] = [
        {"key": k, "label": CLASS_LABELS.get(k, k.title()),
         "color": CLASS_COLORS.get(k, "#38bdf8"), "tracks": int(v)}
        for k, v in _cat_counts.sort_values(ascending=False).items()
    ]''', flags=0)

sub("frame record cat",
    r'cls_name = str\(row\.class_final\)\.lower\(\)',
    'cat = str(row.cat)', flags=0)

sub("frame record fields",
    r'"cls": str\(row\.class_final\),\s*\n(\s*)"cls_color": CLASS_COLORS\.get\(cls_name, "#38bdf8"\),',
    r'''"cls": str(row.class_final),
\1"cat": cat,
\1"cat_label": CLASS_LABELS.get(cat, cat.title()),
\1"cls_color": CLASS_COLORS.get(cat, "#38bdf8"),''')

sub("track cat",
    r'(cls_name = str\(group\["class_final"\]\.iloc\[0\]\))',
    r'\1\n        cat = canon_class(cls_name)', flags=0)

sub("track fields",
    r'"cls": cls_name,\s*\n(\s*)"cls_color": CLASS_COLORS\.get\(cls_name\.lower\(\), "#38bdf8"\),',
    r'''"cls": cls_name,
\1"cat": cat,
\1"cat_label": CLASS_LABELS.get(cat, cat.title()),
\1"cls_color": CLASS_COLORS.get(cat, "#38bdf8"),''')

sub("raw_trks fields",
    r'(raw_trks\.append\(\{\s*\n\s*"id": int\(tid\),\s*\n)(\s*)("cls": cls_name,\s*\n)',
    r'''\1\2\3\2"cat": cat,
\2"cat_label": CLASS_LABELS.get(cat, cat.title()),
\2"t_first": t_arr[0] if t_arr else 0.0,
\2"t_last": t_arr[-1] if t_arr else 0.0,
\2"dist_total": dist_arr[-1] if dist_arr else 0.0,
\2"max_spd": round(float(group["speed_kmh"].max()), 1) if len(group) else 0.0,
''')

# --------------------------------------------------------------------------
# 4. /api/tracks — vehicle directory
# --------------------------------------------------------------------------
API = '''@app.get("/api/tracks")
async def list_tracks(cat: str = "", q: str = "", limit: int = 400):
    """Vehicle directory for the picker: optional category filter + id/class search."""
    wanted = {c.strip().lower() for c in cat.split(",") if c.strip()}
    needle = q.strip().lower()
    rows = []
    for t in RAW_TRACKS_METRIC:
        if wanted and t["cat"] not in wanted:
            continue
        if needle and needle not in str(t["id"]) \\
                and needle not in t["cat_label"].lower() \\
                and needle not in str(t["cls"]).lower():
            continue
        rows.append({k: t[k] for k in
                     ("id", "cls", "cat", "cat_label", "mean_spd", "max_spd",
                      "dist_total", "t_first", "t_last")})
    rows.sort(key=lambda r: r["t_first"])
    return {"count": len(rows), "tracks": rows[:limit]}


@app.get("/", response_class=HTMLResponse)'''

sub("api/tracks", r'@app\.get\("/", response_class=HTMLResponse\)',
    API.replace("\\", "\\\\"), flags=0)

# --------------------------------------------------------------------------
# 5. Client CSS
# --------------------------------------------------------------------------
CSS = '''        .mini-link { color: var(--accent); font-size: 10px; text-decoration: none; margin-left: 8px; }
        .cat-count { margin-left: auto; font-size: 10px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
        .track-search {
            width: 100%; box-sizing: border-box; padding: 6px 8px; border-radius: 6px;
            border: 1px solid var(--border); background: rgba(0,0,0,0.25);
            color: inherit; font-size: 11px; font-family: inherit;
        }
        .track-list {
            max-height: 190px; overflow-y: auto; border: 1px solid var(--border);
            border-radius: 6px; background: rgba(0,0,0,0.2);
        }
        .track-row {
            display: flex; justify-content: space-between; align-items: center; gap: 6px;
            padding: 5px 8px; font-size: 11px; cursor: pointer;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .track-row:hover { background: rgba(56,189,248,0.12); }
        .track-row.active { background: rgba(56,189,248,0.25); font-weight: 700; }
        .track-row-meta { color: var(--text-muted); font-size: 10px; }
        .track-empty { padding: 10px; font-size: 10px; color: var(--text-muted); text-align: center; }
        .track-detail:not(:empty) {
            margin-top: 8px; padding: 8px; border-radius: 6px;
            border: 1px solid var(--border-highlight); background: rgba(56,189,248,0.08); font-size: 11px;
        }
        .detail-title { font-weight: 700; margin-bottom: 6px; }
        .detail-grid { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; }
        .detail-grid span { color: var(--text-muted); }
        .detail-grid b { text-align: right; font-variant-numeric: tabular-nums; }

        .checkbox-group {'''

sub("css", r'        \.checkbox-group \{', CSS.replace("\\", "\\\\"), flags=0)

# --------------------------------------------------------------------------
# 6. Sidebar markup — categories + vehicle picker
# --------------------------------------------------------------------------
SIDEBAR = '''<div class="sidebar-section-title" style="display: flex; justify-content: space-between;">
                        <span>Filter Vehicle Classes</span>
                        <span>
                            <a href="javascript:void(0)" onclick="setAllCategories(true)" class="mini-link">All</a>
                            <a href="javascript:void(0)" onclick="setAllCategories(false)" class="mini-link">None</a>
                        </span>
                    </div>
                    <!-- built from metadata.categories, so LGV / HGV / three-wheeler
                         get a filter instead of being silently dropped -->
                    <div class="checkbox-group" id="categoryGroup"></div>

                    <div class="sidebar-section-title" style="display: flex; justify-content: space-between;">
                        <span>Vehicle Selection</span>
                        <a href="javascript:void(0)" onclick="clearSelection()" class="mini-link">Clear</a>
                    </div>
                    <input type="text" id="trackSearch" class="track-search"
                           placeholder="Search id or class \\u2014 e.g. 1499, bus"
                           oninput="renderTrackList()">
                    <label class="check-item" style="margin: 6px 0;">
                        <input type="checkbox" id="isolateSel" onchange="applySelectionMode()">
                        Show selected vehicle only
                    </label>
                    <div class="track-list" id="trackList"></div>
                    <div class="track-detail" id="trackDetail"></div>
'''

sub("sidebar",
    r'<div class="sidebar-section-title">Filter Vehicle Classes</div>\s*<div class="checkbox-group">.*?</div>\s*\n',
    SIDEBAR.replace("\\", "\\\\"))

sub("heading slider",
    r'<span>Heading:\s*<b id="valBearing">[\d.]+°</b></span>\s*\n(\s*)<input type="range" id="sliderBearing"[^>]*>',
    r'''<span>Heading (+X axis): <b id="valBearing">106.7°</b></span>
\1<input type="range" id="sliderBearing" min="60" max="160" step="0.25" value="106.65" oninput="updateCalib()">
\1</div>
\1<div class="calib-row">
\1<label class="check-item" style="padding: 4px 0;">
\1    <input type="checkbox" id="chkMirror" onchange="updateCalib()"> Mirror frame (debug)
\1</label>''')

# --------------------------------------------------------------------------
# 7. Client JS — projection, filters, picker
# --------------------------------------------------------------------------
sub("js constants",
    r'const BASE_LAT = [\d.]+;\s*\n(\s*)const BASE_LON = [\d.]+;\s*\n\s*let calibBearing = [\d.]+;',
    r'''// Anchor = the OSM-fitted junction. calibBearing is the compass bearing of
\1// the video frame's +X axis; frame +Y points DOWN the image, i.e. along
\1// bearing + 90 -- that is what keeps traces un-mirrored and square with
\1// the carriageways.
\1const BASE_LAT = 18.566057;
\1const BASE_LON = 73.771440;
\1const FRAME_X0 = 9.99;
\1const FRAME_Y0 = 68.32;
\1const DEFAULT_HEADING = 106.65;
\1const M_PER_DEG_LAT = 111320.0;
\1const M_PER_DEG_LON = 111320.0 * Math.cos(BASE_LAT * Math.PI / 180.0);

\1let categoryDefs = [];
\1let isolateSelected = false;
\1let calibMirror = false;
\1let calibBearing = DEFAULT_HEADING;''')

PROJ = '''        // Image frame (+X right, +Y down) -> WGS84.
        // +X lies along calibBearing, so +Y lies along calibBearing + 90.
        function projectPoint(xRaw, yRaw) {
            let xm = (xRaw - FRAME_X0) * calibScale;
            const ym = (yRaw - FRAME_Y0) * calibScale;
            if (calibMirror) xm = -xm;

            const h1 = calibBearing * Math.PI / 180.0;
            const h2 = h1 + Math.PI / 2.0;

            const east  = xm * Math.sin(h1) + ym * Math.sin(h2);
            const north = xm * Math.cos(h1) + ym * Math.cos(h2);

            return [
                BASE_LAT + (north + calibLatShift) / M_PER_DEG_LAT,
                BASE_LON + (east + calibLonShift) / M_PER_DEG_LON
            ];
        }

        function projectTrack(trk) {
            const out = [];
            for (let i = 0; i < trk.x_m.length; i++) {
                out.push(projectPoint(trk.x_m[i], trk.y_m[i]));
            }
            return out;
        }

        // Drawn when its category is ticked and -- if "show selected vehicle
        // only" is on -- when it is the selected track.
        function isTrackVisible(trk, selectedClasses) {
            if (isolateSelected) return selectedTrackId === trk.id;
            return selectedClasses.has((trk.cat || 'other').toLowerCase());
        }

        function recomputeAndRenderTraces() {
            tracePolyLayers.forEach(l => map.removeLayer(l));
            tracePolyLayers = [];

            const selectedClasses = getSelectedCategories();

            for (let trk of rawMetricTracks) {
                // always reproject, so the live head-trace stays in sync even
                // for tracks whose full history is currently hidden
                trk.projected_coords = projectTrack(trk);

                if (!isTrackVisible(trk, selectedClasses)) continue;

                if (activeColorMode !== 'active') {'''

sub("js projection",
    r'function recomputeAndRenderTraces\(\) \{\s*\n\s*tracePolyLayers\.forEach.*?if \(activeColorMode !== \'active\'\) \{',
    PROJ.replace("\\", "\\\\"))

sub("js calib read",
    r"(calibScale = parseFloat\(document\.getElementById\('sliderScale'\)\.value\) / 100\.0;)",
    r"\1\n            calibMirror = document.getElementById('chkMirror').checked;", flags=0)

sub("js calib reset",
    r"document\.getElementById\('sliderBearing'\)\.value = [\d.]+;",
    "document.getElementById('sliderBearing').value = DEFAULT_HEADING;\n"
    "            document.getElementById('chkMirror').checked = false;", flags=0)

PICKER = '''        function getSelectedCategories() {
            const set = new Set();
            document.querySelectorAll('#categoryGroup input:checked').forEach(cb => {
                set.add(cb.value.toLowerCase());
            });
            return set;
        }

        const CATEGORY_ICONS = {
            car: '\\ud83d\\ude97', motorcycle: '\\ud83c\\udfcd\\ufe0f', auto: '\\ud83d\\udefa',
            bus: '\\ud83d\\ude8c', hgv: '\\ud83d\\ude9a', lgv: '\\ud83d\\ude90',
            cyclist: '\\ud83d\\udeb2', pedestrian: '\\ud83d\\udeb6', other: '\\u2753'
        };

        function renderCategories() {
            document.getElementById('categoryGroup').innerHTML = categoryDefs.map(c => `
                <label class="check-item">
                    <input type="checkbox" value="${c.key}" checked onchange="filterCategory()">
                    <span style="color:${c.color};">${CATEGORY_ICONS[c.key] || '\\u2753'}</span>
                    ${c.label}
                    <span class="cat-count">${c.tracks}</span>
                </label>`).join('');
        }

        function setAllCategories(on) {
            document.querySelectorAll('#categoryGroup input').forEach(cb => { cb.checked = on; });
            filterCategory();
        }

        function filterCategory() {
            renderTrackList();
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        // ---- vehicle picker ------------------------------------------------
        function renderTrackList() {
            const cats = getSelectedCategories();
            const q = document.getElementById('trackSearch').value.trim().toLowerCase();
            const rows = rawMetricTracks
                .filter(t => cats.has((t.cat || 'other').toLowerCase()))
                .filter(t => !q || String(t.id).includes(q)
                          || (t.cat_label || '').toLowerCase().includes(q)
                          || String(t.cls).toLowerCase().includes(q))
                .sort((a, b) => a.t_first - b.t_first);

            const list = document.getElementById('trackList');
            if (!rows.length) {
                list.innerHTML = '<div class="track-empty">No vehicles match.</div>';
                return;
            }
            list.innerHTML = rows.slice(0, 300).map(t => `
                <div class="track-row ${selectedTrackId === t.id ? 'active' : ''}" onclick="selectTrack(${t.id})">
                    <span>${CATEGORY_ICONS[t.cat] || '\\u2753'} #${t.id}</span>
                    <span class="track-row-meta">${t.cat_label} \\u00b7 ${t.mean_spd} km/h</span>
                </div>`).join('')
                + (rows.length > 300
                    ? `<div class="track-empty">+${rows.length - 300} more \\u2014 refine the search.</div>`
                    : '');
        }

        function renderTrackDetail() {
            const box = document.getElementById('trackDetail');
            const t = rawMetricTracks.find(v => v.id === selectedTrackId);
            if (!t) { box.innerHTML = ''; return; }
            box.innerHTML = `
                <div class="detail-title">${CATEGORY_ICONS[t.cat] || '\\u2753'} Track #${t.id} \\u2014 ${t.cat_label}</div>
                <div class="detail-grid">
                    <span>Mean speed</span><b>${t.mean_spd} km/h</b>
                    <span>Peak speed</span><b>${t.max_spd} km/h</b>
                    <span>Distance</span><b>${t.dist_total} m</b>
                    <span>On scene</span><b>${t.t_first.toFixed(1)}s \\u2192 ${t.t_last.toFixed(1)}s</b>
                    <span>Raw class</span><b>${t.cls}</b>
                </div>`;
        }

        function applySelectionMode() {
            isolateSelected = document.getElementById('isolateSel').checked;
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        function clearSelection() {
            selectedTrackId = null;
            document.getElementById('isolateSel').checked = false;
            isolateSelected = false;
            if (typeof selectedHeader !== 'undefined' && selectedHeader) {
                selectedHeader.innerText = 'All vehicles';
            }
            renderTrackList();
            renderTrackDetail();
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }'''

sub("js picker",
    r"function getSelectedCategories\(\) \{.*?function filterCategory\(\) \{.*?\n        \}",
    PICKER.replace("\\", "\\\\"))

sub("js selectTrack",
    r"function selectTrack\(tid\) \{\s*\n\s*selectedTrackId = tid;",
    '''function selectTrack(tid) {
            selectedTrackId = (selectedTrackId === tid) ? null : tid;
            if (selectedTrackId === null) { clearSelection(); return; }
            renderTrackList();
            renderTrackDetail();
            setTimeout(() => {
                const trk = rawMetricTracks.find(t => t.id === selectedTrackId);
                if (trk && trk.projected_coords && trk.projected_coords.length) {
                    map.fitBounds(L.latLngBounds(trk.projected_coords).pad(0.35));
                }
            }, 0);
            tid = selectedTrackId;''')

sub("js frame filter",
    r"const objs = rawObjs\.filter\(o => selectedClasses\.has\(o\.cls\.toLowerCase\(\)\)\);",
    """const objs = rawObjs.filter(o => isolateSelected
                ? o.id === selectedTrackId
                : selectedClasses.has((o.cat || 'other').toLowerCase()));""", flags=0)

sub("js initData",
    r"(rawMetricTracks = data\.raw_metric;)",
    r"""\1
                categoryDefs = (data.metadata || {}).categories || [];
                renderCategories();
                renderTrackList();""", flags=0)

PATH.write_text(src)
print(f"[+] applied : {', '.join(applied)}")
if skipped:
    print(f"[!] NOT FOUND (check manually): {', '.join(skipped)}")
print(f"[\u2713] wrote {PATH}")
