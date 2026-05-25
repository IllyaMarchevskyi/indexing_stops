#!/usr/bin/env python3
"""Interactive reviewer for EasyWay routes against OSM stop geometries."""

from __future__ import annotations

import argparse
import curses
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from match_stops import (
    PYPPETEER_BROWSER_MANAGER,
    extract_stop_ids,
    load_decisions,
    normalize_name,
    open_url_in_browser,
    parse_point_wkt,
    read_csv_rows,
    save_decisions,
    similarity,
)


TRANSPORT_LABELS = {
    "bus": "автобус",
    "trol": "тролейбус",
    "tram": "трамвай",
    "metro": "метро",
}

CALENDAR_ORDER = {
    "All Week": 0,
    "Weekdays": 1,
    "Weekends": 2,
}

DIRECTION_ORDER = {
    "forward": 0,
    "backward": 1,
}

ROUTE_PREVIEW_COLORS = {
    ("forward", "Weekdays"): "#1d4ed8",
    ("forward", "Weekends"): "#60a5fa",
    ("backward", "Weekdays"): "#15803d",
    ("backward", "Weekends"): "#4ade80",
    ("forward", "All Week"): "#2563eb",
    ("backward", "All Week"): "#16a34a",
}


@dataclass(frozen=True)
class RouteRow:
    route_row_id: int
    calendar: str
    route_id: str
    stop_id: str
    transport: str
    route: str
    direction: str
    stop_name: str
    index: str
    schedules: str


@dataclass(frozen=True)
class RouteGroup:
    key: str
    route_id: str
    route: str
    transport: str
    rows: Tuple[int, ...]


@dataclass(frozen=True)
class StopView:
    stop_id: str
    stop_name: str
    route_row_ids: Tuple[int, ...]
    direction: str
    calendar: str
    index: int
    decision_key: Optional[str]
    assigned_osm_name: str
    geometry: str
    status: str
    previous_geometry: str


@dataclass(frozen=True)
class OsmCandidate:
    osm_row_id: int
    osm_name: str
    geometry: str
    score: float
    prev_distance_m: Optional[float]
    assigned_elsewhere: bool
    current: bool


def transport_label(raw: str) -> str:
    return TRANSPORT_LABELS.get(raw, raw)


def route_sort_key(group: RouteGroup) -> Tuple[str, str, str]:
    return (transport_label(group.transport), group.route, group.route_id)


def row_sort_key(row: RouteRow) -> Tuple[int, int, int, int]:
    try:
        index_value = int(row.index)
    except ValueError:
        index_value = 10**9
    return (
        DIRECTION_ORDER.get(row.direction, 9),
        CALENDAR_ORDER.get(row.calendar, 9),
        index_value,
        row.route_row_id,
    )


def route_link(route_id: str) -> str:
    return f"https://www.eway.in.ua/ua/cities/kyiv/routes/{route_id}"


def load_reviewed_routes(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}
    return {}


def save_reviewed_routes(path: Path, reviewed: Dict[str, Dict[str, object]]) -> None:
    path.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8")


def build_route_rows(route_rows: Sequence[Dict[str, str]]) -> List[RouteRow]:
    built: List[RouteRow] = []
    for route_row_id, row in enumerate(route_rows):
        built.append(
            RouteRow(
                route_row_id=route_row_id,
                calendar=row.get("calendar", "").strip(),
                route_id=row.get("route_id", "").strip(),
                stop_id=row.get("stop_id", "").strip(),
                transport=row.get("transport", "").strip(),
                route=row.get("route", "").strip(),
                direction=row.get("direction", "").strip(),
                stop_name=row.get("stop_name", "").strip(),
                index=row.get("index", "").strip(),
                schedules=row.get("schedules", "").strip(),
            )
        )
    return built


def build_route_groups(rows: Sequence[RouteRow]) -> List[RouteGroup]:
    grouped: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for row in rows:
        grouped[(row.transport, row.route, row.route_id)].append(row.route_row_id)
    groups = [
        RouteGroup(
            key=f"{transport}|{route_id}",
            route_id=route_id,
            route=route,
            transport=transport,
            rows=tuple(sorted(row_ids)),
        )
        for (transport, route, route_id), row_ids in grouped.items()
    ]
    groups.sort(key=route_sort_key)
    return groups


def decision_maps(decisions: Dict[str, Dict[str, object]]) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    row_to_decision: Dict[int, str] = {}
    stop_id_to_rows: Dict[str, List[int]] = defaultdict(list)
    for decision_key, decision in decisions.items():
        for route_row_id in decision.get("route_row_ids", []):
            route_row_id_int = int(route_row_id)
            row_to_decision[route_row_id_int] = decision_key
        for stop_id in decision.get("stop_ids", []):
            stop_id_to_rows[str(stop_id)].extend(int(route_row_id) for route_row_id in decision.get("route_row_ids", []))
    return row_to_decision, stop_id_to_rows


def distance_meters(left_geometry: str, right_geometry: str) -> Optional[float]:
    left = parse_point_wkt(left_geometry)
    right = parse_point_wkt(right_geometry)
    if left is None or right is None:
        return None
    lon1, lat1 = left
    lon2, lat2 = right
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_stop_views(
    group: RouteGroup,
    all_rows: Sequence[RouteRow],
    decisions: Dict[str, Dict[str, object]],
) -> List[StopView]:
    row_lookup = {row.route_row_id: row for row in all_rows}
    row_to_decision, _ = decision_maps(decisions)
    grouped: Dict[str, List[RouteRow]] = defaultdict(list)
    for route_row_id in group.rows:
        grouped[row_lookup[route_row_id].stop_id].append(row_lookup[route_row_id])

    stop_views: List[StopView] = []
    for stop_id, rows in grouped.items():
        rows_sorted = sorted(rows, key=row_sort_key)
        representative = rows_sorted[0]
        route_row_ids = tuple(sorted(row.route_row_id for row in rows_sorted))
        assigned_decisions = {row_to_decision[row.route_row_id] for row in rows_sorted if row.route_row_id in row_to_decision}
        decision_key = None
        assigned_osm_name = ""
        geometry = ""
        status = "missing"
        if len(assigned_decisions) == 1:
            decision_key = next(iter(assigned_decisions))
            decision = decisions.get(decision_key, {})
            assigned_osm_name = str(decision.get("osm_name", ""))
            geometry = str(decision.get("geometry", ""))
            status = "matched"
        elif len(assigned_decisions) > 1:
            status = "inconsistent"

        previous_geometry = ""
        same_chain = [row_lookup[row_id] for row_id in group.rows if row_lookup[row_id].direction == representative.direction and row_lookup[row_id].calendar == representative.calendar]
        same_chain.sort(key=row_sort_key)
        for chain_row in same_chain:
            if chain_row.route_row_id == representative.route_row_id:
                break
            previous_key = row_to_decision.get(chain_row.route_row_id)
            if previous_key:
                previous_geometry = str(decisions.get(previous_key, {}).get("geometry", ""))

        stop_views.append(
            StopView(
                stop_id=stop_id,
                stop_name=representative.stop_name,
                route_row_ids=route_row_ids,
                direction=representative.direction,
                calendar=representative.calendar,
                index=int(representative.index) if representative.index.isdigit() else 10**9,
                decision_key=decision_key,
                assigned_osm_name=assigned_osm_name,
                geometry=geometry,
                status=status,
                previous_geometry=previous_geometry,
            )
        )

    stop_views.sort(key=lambda item: (DIRECTION_ORDER.get(item.direction, 9), CALENDAR_ORDER.get(item.calendar, 9), item.index, item.stop_name.lower(), item.stop_id))
    return stop_views


def build_osm_candidates(
    stop_view: StopView,
    osm_rows: Sequence[Dict[str, str]],
    decisions: Dict[str, Dict[str, object]],
) -> List[OsmCandidate]:
    row_to_decision, _ = decision_maps(decisions)
    assigned_osm_keys = set(decisions.keys())
    candidates: List[OsmCandidate] = []
    for osm_row_id, row in enumerate(osm_rows):
        osm_name = row.get("name", "").strip()
        geometry = row.get("geometry", "").strip()
        score = similarity(stop_view.stop_name, osm_name)
        prev_distance = distance_meters(stop_view.previous_geometry, geometry) if stop_view.previous_geometry else None
        current = stop_view.decision_key == str(osm_row_id)
        assigned_elsewhere = str(osm_row_id) in assigned_osm_keys and not current
        candidates.append(
            OsmCandidate(
                osm_row_id=osm_row_id,
                osm_name=osm_name,
                geometry=geometry,
                score=score,
                prev_distance_m=prev_distance,
                assigned_elsewhere=assigned_elsewhere,
                current=current,
            )
        )

    def sort_key(candidate: OsmCandidate) -> Tuple[float, float, str]:
        distance_value = candidate.prev_distance_m if candidate.prev_distance_m is not None else float("inf")
        return (-candidate.score, distance_value, normalize_name(candidate.osm_name))

    candidates.sort(key=sort_key)

    selected: List[OsmCandidate] = []
    seen_ids = set()
    for candidate in candidates:
        selected.append(candidate)
        seen_ids.add(candidate.osm_row_id)
        if len(selected) >= 25 and candidate.score < 0.15:
            break
        if len(selected) >= 40:
            break
    if stop_view.decision_key is not None:
        current_id = int(stop_view.decision_key)
        if current_id not in seen_ids and 0 <= current_id < len(osm_rows):
            row = osm_rows[current_id]
            selected.append(
                OsmCandidate(
                    osm_row_id=current_id,
                    osm_name=row.get("name", "").strip(),
                    geometry=row.get("geometry", "").strip(),
                    score=similarity(stop_view.stop_name, row.get("name", "").strip()),
                    prev_distance_m=distance_meters(stop_view.previous_geometry, row.get("geometry", "").strip()) if stop_view.previous_geometry else None,
                    assigned_elsewhere=False,
                    current=True,
                )
            )
            selected.sort(key=sort_key)
    return selected


def update_decision_stop_ids(decision: Dict[str, object], route_rows: Sequence[RouteRow]) -> None:
    stop_ids = extract_stop_ids({row.route_row_id: row for row in route_rows}, [int(route_row_id) for route_row_id in decision.get("route_row_ids", [])])
    decision["stop_ids"] = stop_ids
    decision["stop_id"] = stop_ids[0] if len(stop_ids) == 1 else None


def reassign_stop_id(
    stop_id: str,
    stop_name: str,
    target_osm_row_id: int,
    route_rows: Sequence[RouteRow],
    osm_rows: Sequence[Dict[str, str]],
    decisions: Dict[str, Dict[str, object]],
) -> None:
    affected_route_row_ids = sorted(row.route_row_id for row in route_rows if row.stop_id == stop_id)
    if not affected_route_row_ids:
        return

    for decision_key, decision in decisions.items():
        route_row_ids = [int(route_row_id) for route_row_id in decision.get("route_row_ids", [])]
        filtered = [route_row_id for route_row_id in route_row_ids if route_row_id not in affected_route_row_ids]
        if len(filtered) != len(route_row_ids):
            decision["route_row_ids"] = filtered
            update_decision_stop_ids(decision, route_rows)
            if not filtered:
                decision["status"] = "skipped"
                decision["stop_id"] = None
                decision["stop_ids"] = []

    target_key = str(target_osm_row_id)
    target_decision = decisions.get(target_key)
    osm_row = osm_rows[target_osm_row_id]
    if target_decision is None:
        target_decision = {
            "status": "matched",
            "osm_name": osm_row.get("name", "").strip(),
            "geometry": osm_row.get("geometry", "").strip(),
            "candidate_name": stop_name,
            "route_row_ids": [],
        }
        decisions[target_key] = target_decision

    merged = sorted(set(int(route_row_id) for route_row_id in target_decision.get("route_row_ids", [])) | set(affected_route_row_ids))
    target_decision["status"] = "matched"
    target_decision["osm_name"] = osm_row.get("name", "").strip()
    target_decision["geometry"] = osm_row.get("geometry", "").strip()
    target_decision.setdefault("candidate_name", stop_name)
    target_decision["route_row_ids"] = merged
    update_decision_stop_ids(target_decision, route_rows)


def route_summary(group: RouteGroup, stop_views: Sequence[StopView]) -> str:
    total = len(stop_views)
    matched = sum(1 for item in stop_views if item.status == "matched")
    missing = sum(1 for item in stop_views if item.status == "missing")
    inconsistent = sum(1 for item in stop_views if item.status == "inconsistent")
    return (
        f"{transport_label(group.transport)} {group.route} | route_id {group.route_id} | "
        f"зупинок {total} | mapped {matched} | missing {missing} | conflict {inconsistent}"
    )


def write_route_preview_map(
    output_path: Path,
    group: RouteGroup,
    route_rows: Sequence[RouteRow],
    stop_views: Sequence[StopView],
    selected_stop_id: Optional[str] = None,
    candidate: Optional[OsmCandidate] = None,
) -> Optional[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_lookup = {row.route_row_id: row for row in route_rows}
    view_by_row_id: Dict[int, StopView] = {}
    for stop_view in stop_views:
        for route_row_id in stop_view.route_row_ids:
            view_by_row_id[route_row_id] = stop_view

    polylines: List[Dict[str, object]] = []
    markers: List[Dict[str, object]] = []
    guessed_markers: List[Dict[str, object]] = []
    all_coords: List[Tuple[float, float]] = []

    by_variant: Dict[Tuple[str, str], List[RouteRow]] = defaultdict(list)
    for route_row_id in group.rows:
        by_variant[(row_lookup[route_row_id].direction, row_lookup[route_row_id].calendar)].append(row_lookup[route_row_id])

    for (direction, calendar), rows in by_variant.items():
        rows_sorted = sorted(rows, key=row_sort_key)
        latlngs: List[List[float]] = []
        for row in rows_sorted:
            stop_view = view_by_row_id.get(row.route_row_id)
            geometry = stop_view.geometry if stop_view is not None else ""
            coords = parse_point_wkt(geometry)
            if coords is None:
                if latlngs:
                    polylines.append(
                        {
                            "direction": direction,
                            "calendar": calendar,
                            "color": ROUTE_PREVIEW_COLORS.get((direction, calendar), "#6b7280"),
                            "latlngs": latlngs,
                        }
                    )
                    latlngs = []
                continue
            lon, lat = coords
            all_coords.append((lat, lon))
            latlngs.append([lat, lon])
            marker_color = "#f97316" if row.stop_id == selected_stop_id else ("#2563eb" if direction == "forward" else "#16a34a")
            if stop_view is not None and stop_view.status != "matched":
                marker_color = "#dc2626"
            markers.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "color": marker_color,
                    "label": f"{row.direction} #{row.index} | {row.stop_name} | stop_id {row.stop_id}",
                }
            )
        if latlngs:
            polylines.append(
                {
                    "direction": direction,
                    "calendar": calendar,
                    "color": ROUTE_PREVIEW_COLORS.get((direction, calendar), "#6b7280"),
                    "latlngs": latlngs,
                }
            )

    if candidate is not None:
        coords = parse_point_wkt(candidate.geometry)
        if coords is not None:
            lon, lat = coords
            guessed_markers.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "label": f"Кандидат: {candidate.osm_name}",
                }
            )
            all_coords.append((lat, lon))

    center_lat, center_lon = (50.4501, 30.5234)
    if all_coords:
        center_lat = sum(lat for lat, _ in all_coords) / len(all_coords)
        center_lon = sum(lon for _, lon in all_coords) / len(all_coords)

    html = f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>Route Preview</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .info {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1000;
      width: min(480px, calc(100vw - 24px));
      background: rgba(255, 255, 255, 0.95);
      padding: 12px 14px;
      border-radius: 10px;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .title {{ font-weight: 700; margin-bottom: 6px; }}
    .legend {{ margin-top: 8px; display: grid; gap: 4px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
  </style>
</head>
<body>
  <div class="info">
    <div class="title">{transport_label(group.transport)} {group.route} (route_id {group.route_id})</div>
    <div>{route_summary(group, stop_views)}</div>
    <div class="legend">
      <div class="legend-row"><span class="dot" style="background:#2563eb"></span>Прямий</div>
      <div class="legend-row"><span class="dot" style="background:#16a34a"></span>Зворотний</div>
      <div class="legend-row"><span class="dot" style="background:#dc2626"></span>Немає прив'язки</div>
      <div class="legend-row"><span class="dot" style="background:#f97316"></span>Поточна зупинка</div>
      <div class="legend-row"><span class="dot" style="background:#a855f7"></span>Автовгаданий кандидат</div>
    </div>
  </div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([{center_lat}, {center_lon}], 13);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    const polylines = {json.dumps(polylines, ensure_ascii=False)};
    const markers = {json.dumps(markers, ensure_ascii=False)};
    const guessed = {json.dumps(guessed_markers, ensure_ascii=False)};
    const bounds = [];
    for (const item of polylines) {{
      if (!item.latlngs.length) continue;
      const line = L.polyline(item.latlngs, {{
        color: item.color,
        weight: 4,
        opacity: 0.9,
      }}).addTo(map).bindPopup(`${{item.direction}} | ${{item.calendar}}`);
      bounds.push(...item.latlngs);
    }}
    for (const item of markers) {{
      L.circleMarker([item.lat, item.lon], {{
        radius: 6,
        color: item.color,
        fillColor: item.color,
        fillOpacity: 0.92,
        weight: 1,
      }}).addTo(map).bindPopup(item.label);
      bounds.push([item.lat, item.lon]);
    }}
    for (const item of guessed) {{
      L.circleMarker([item.lat, item.lon], {{
        radius: 8,
        color: '#a855f7',
        fillColor: '#a855f7',
        fillOpacity: 0.35,
        weight: 2,
      }}).addTo(map).bindPopup(item.label);
      bounds.push([item.lat, item.lon]);
    }}
    if (bounds.length) {{
      map.fitBounds(bounds, {{ padding: [24, 24] }});
    }}
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def open_route_pages(group: RouteGroup, preview_map_path: Path) -> None:
    open_url_in_browser(route_link(group.route_id), None, target_kind="eway")
    open_url_in_browser(preview_map_path.resolve().as_uri(), None, target_kind="preview")


def draw_menu(stdscr: "curses._CursesWindow", title: str, lines: Sequence[str], current: int, footer: str) -> None:
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()
    width = max(10, max_x - 1)
    stdscr.addnstr(0, 0, title, width, curses.A_BOLD)
    visible_height = max(1, max_y - 3)
    start = 0
    if current >= visible_height:
        start = current - visible_height + 1
    for idx, line in enumerate(lines[start : start + visible_height], start=start):
        attr = curses.A_REVERSE if idx == current else 0
        stdscr.addnstr(1 + idx - start, 0, line, width, attr)
    stdscr.addnstr(max_y - 1, 0, footer, width, curses.A_DIM)
    stdscr.refresh()


def choose_candidate_tui(
    stdscr: "curses._CursesWindow",
    group: RouteGroup,
    stop_view: StopView,
    candidates: Sequence[OsmCandidate],
) -> Optional[OsmCandidate]:
    current = 0
    while True:
        lines = []
        for idx, candidate in enumerate(candidates):
            parts = [f"[{candidate.score:.3f}] {candidate.osm_name}", f"osm_row_id {candidate.osm_row_id}"]
            if candidate.prev_distance_m is not None and math.isfinite(candidate.prev_distance_m):
                parts.append(f"prev {candidate.prev_distance_m:.0f}m")
            if candidate.current:
                parts.append("CURRENT")
            elif candidate.assigned_elsewhere:
                parts.append("already used")
            if idx == 0:
                parts.append("AUTO?")
            lines.append(" | ".join(parts))

        draw_menu(
            stdscr,
            title=f"Кандидати для {stop_view.stop_name} | stop_id {stop_view.stop_id} | маршрут {group.route}",
            lines=lines,
            current=current,
            footer="Enter: вибрати | g: top auto | b: назад | q: вихід",
        )
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (ord("b"), ord("h")):
            return None
        if key in (curses.KEY_UP, ord("k")) and current > 0:
            current -= 1
        elif key in (curses.KEY_DOWN, ord("j")) and current + 1 < len(candidates):
            current += 1
        elif key in (10, 13, curses.KEY_ENTER):
            return candidates[current]
        elif key == ord("g") and candidates:
            return candidates[0]


def review_route_tui(
    stdscr: "curses._CursesWindow",
    group: RouteGroup,
    route_rows: Sequence[RouteRow],
    osm_rows: Sequence[Dict[str, str]],
    decisions_path: Path,
    decisions: Dict[str, Dict[str, object]],
    preview_map_path: Path,
) -> Tuple[str, Dict[str, Dict[str, object]]]:
    current = 0
    decisions_local = decisions
    open_route_pages(group, preview_map_path)
    refresh_preview = True
    candidate_cache: Dict[Tuple[str, str, str], List[OsmCandidate]] = {}
    selected_auto_label = ""

    def get_candidates(stop_view: StopView) -> List[OsmCandidate]:
        cache_key = (
            stop_view.stop_id,
            stop_view.decision_key or "",
            stop_view.previous_geometry,
        )
        cached = candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        computed = build_osm_candidates(stop_view, osm_rows, decisions_local)
        candidate_cache[cache_key] = computed
        return computed

    while True:
        stop_views = build_stop_views(group, route_rows, decisions_local)
        if not stop_views:
            return "back", decisions_local
        current = min(current, len(stop_views) - 1)
        selected = stop_views[current]
        if refresh_preview:
            auto_candidate = None
            selected_auto_label = ""
            if selected.status != "matched":
                candidates_for_selected = get_candidates(selected)
                if candidates_for_selected:
                    auto_candidate = candidates_for_selected[0]
                    selected_auto_label = auto_candidate.osm_name
            write_route_preview_map(preview_map_path, group, route_rows, stop_views, selected_stop_id=selected.stop_id, candidate=auto_candidate)
            open_url_in_browser(preview_map_path.resolve().as_uri(), None, target_kind="preview")
            refresh_preview = False

        lines = []
        for item in stop_views:
            marker = " "
            if item.status == "missing":
                marker = "!"
            elif item.status == "inconsistent":
                marker = "?"
            elif item.stop_id == selected.stop_id:
                marker = ">"
            mapped = item.assigned_osm_name or "нема OSM"
            if item.status != "matched" and item.stop_id == selected.stop_id and selected_auto_label:
                mapped = f"AUTO? {selected_auto_label}"
            lines.append(
                f"{marker} {item.direction[:1].upper()} #{item.index} | {item.stop_name} | stop_id {item.stop_id} | {mapped}"
            )

        draw_menu(
            stdscr,
            title=route_summary(group, stop_views),
            lines=lines,
            current=current,
            footer="Enter/e: редагувати | g: top auto | o: відкрити eway | c: підтвердити маршрут | b: назад | q: вихід",
        )
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return "quit", decisions_local
        if key in (ord("b"), ord("h")):
            return "back", decisions_local
        if key in (curses.KEY_UP, ord("k")) and current > 0:
            current -= 1
            continue
        if key in (curses.KEY_DOWN, ord("j")) and current + 1 < len(stop_views):
            current += 1
            continue
        if key == ord("o"):
            open_url_in_browser(route_link(group.route_id), None, target_kind="eway")
            continue
        if key == ord("c"):
            return "confirm", decisions_local
        if key in (10, 13, curses.KEY_ENTER, ord("e"), ord("g")):
            candidates = get_candidates(selected)
            if not candidates:
                continue
            chosen = candidates[0] if key == ord("g") else choose_candidate_tui(stdscr, group, selected, candidates)
            if chosen is None:
                continue
            reassign_stop_id(
                stop_id=selected.stop_id,
                stop_name=selected.stop_name,
                target_osm_row_id=chosen.osm_row_id,
                route_rows=route_rows,
                osm_rows=osm_rows,
                decisions=decisions_local,
            )
            save_decisions(decisions_path, decisions_local)
            candidate_cache.clear()
            selected_auto_label = ""
            refresh_preview = True


def run_tui(
    route_groups: Sequence[RouteGroup],
    route_rows: Sequence[RouteRow],
    osm_rows: Sequence[Dict[str, str]],
    decisions_path: Path,
    reviewed_path: Path,
    preview_map_path: Path,
) -> int:
    reviewed = load_reviewed_routes(reviewed_path)
    decisions = load_decisions(decisions_path)
    route_summary_cache: Dict[str, str] = {}

    def get_route_summary(group: RouteGroup) -> str:
        cached = route_summary_cache.get(group.key)
        if cached is not None:
            return cached
        summary = route_summary(group, build_stop_views(group, route_rows, decisions))
        route_summary_cache[group.key] = summary
        return summary

    def run(stdscr: "curses._CursesWindow") -> int:
        curses.curs_set(0)
        current = 0
        while True:
            active_groups = [group for group in route_groups if group.key not in reviewed]
            if not active_groups:
                draw_menu(
                    stdscr,
                    title="Неперевірених маршрутів не залишилось.",
                    lines=[],
                    current=0,
                    footer="q: вихід",
                )
                if stdscr.getch() in (ord("q"), 27):
                    return 0
                continue

            current = min(current, len(active_groups) - 1)
            lines = [get_route_summary(group) for group in active_groups]

            draw_menu(
                stdscr,
                title="Маршрути для перевірки",
                lines=lines,
                current=current,
                footer="Enter: перевірити маршрут | q: вихід",
            )
            key = stdscr.getch()
            if key in (ord("q"), 27):
                return 0
            if key in (curses.KEY_UP, ord("k")) and current > 0:
                current -= 1
                continue
            if key in (curses.KEY_DOWN, ord("j")) and current + 1 < len(active_groups):
                current += 1
                continue
            if key not in (10, 13, curses.KEY_ENTER):
                continue

            group = active_groups[current]
            action, updated_decisions = review_route_tui(
                stdscr=stdscr,
                group=group,
                route_rows=route_rows,
                osm_rows=osm_rows,
                decisions_path=decisions_path,
                decisions=decisions,
                preview_map_path=preview_map_path,
            )
            decisions.clear()
            decisions.update(updated_decisions)
            if action == "quit":
                return 0
            if action == "confirm":
                reviewed[group.key] = {
                    "route_id": group.route_id,
                    "route": group.route,
                    "transport": group.transport,
                }
                save_reviewed_routes(reviewed_path, reviewed)
                current = min(current, max(0, len(active_groups) - 2))
            route_summary_cache.pop(group.key, None)
        return 0

    return curses.wrapper(run)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", default="easyway/easyway_routes.csv", help="Path to EasyWay routes CSV.")
    parser.add_argument("--osm", default="osm/osm_stops.csv", help="Path to OSM stops CSV.")
    parser.add_argument("--decisions", default="stop-match-decisions.json", help="Path to stop decisions JSON.")
    parser.add_argument("--reviewed-routes", default="route-review-decisions.json", help="Path to reviewed route decisions JSON.")
    parser.add_argument("--preview-map", default="current_route_preview.html", help="HTML file overwritten with the current route preview.")
    parser.add_argument("--pyppeteer-executable-path", default="", help="Optional browser executable path for pyppeteer.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    PYPPETEER_BROWSER_MANAGER.executable_path = args.pyppeteer_executable_path or None
    route_rows_raw = read_csv_rows(Path(args.routes))
    osm_rows = read_csv_rows(Path(args.osm))
    if not route_rows_raw:
        raise SystemExit(f"{args.routes} is empty")
    if not osm_rows:
        raise SystemExit(f"{args.osm} is empty")
    route_rows = build_route_rows(route_rows_raw)
    route_groups = build_route_groups(route_rows)
    return run_tui(
        route_groups=route_groups,
        route_rows=route_rows,
        osm_rows=osm_rows,
        decisions_path=Path(args.decisions),
        reviewed_path=Path(args.reviewed_routes),
        preview_map_path=Path(args.preview_map),
    )


if __name__ == "__main__":
    raise SystemExit(main())
