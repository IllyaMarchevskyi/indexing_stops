#!/usr/bin/env python3
"""Interactive matcher that transfers OSM stop geometries to kyiv-routes rows."""

from __future__ import annotations

import argparse
import atexit
import asyncio
import csv
import curses
import difflib
import json
import re
import sys
import textwrap
import threading
import unicodedata
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from pyppeteer import launch
except ImportError:
    launch = None  # type: ignore[assignment]


QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "’": "'",
        "`": "'",
        "–": "-",
        "—": "-",
    }
)

ABBREVIATIONS = {
    "вул": "вулиця",
    "вулиця": "вулиця",
    "просп": "проспект",
    "проспект": "проспект",
    "бул": "бульвар",
    "бульв": "бульвар",
    "бульвар": "бульвар",
    "пл": "площа",
    "площа": "площа",
    "пров": "провулок",
    "провулок": "провулок",
    "жк": "житловий комплекс",
    "ж/к": "житловий комплекс",
    "ст": "станція",
    "стм": "станція метро",
    "метро": "метро",
    "майд": "майдан",
    "наб": "набережна",
}

PHRASE_NORMALIZATIONS = {
    "автоцентр": "автомобільний центр",
}


@dataclass(frozen=True)
class OsmStop:
    osm_row_id: int
    name: str
    geometry: str
    normalized: str


@dataclass(frozen=True)
class RouteContext:
    route_row_id: int
    stop_id: str
    stop_name: str
    route_name: str
    route_id: str
    direction: str
    transport: str
    prev_stop_name: str
    next_stop_name: str


@dataclass(frozen=True)
class ContextOption:
    route_row_ids: Tuple[int, ...]
    context: RouteContext


@dataclass(frozen=True)
class DisplayRow:
    text: str
    attr: int = 0
    option_index: Optional[int] = None


@dataclass(frozen=True)
class InteractionResult:
    status: str
    search_name: str
    selected_name: Optional[str] = None
    selected_route_row_ids: Optional[List[int]] = None


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", (name or "").strip().lower())
    text = text.translate(QUOTE_TRANSLATION)
    text = text.replace("ж/к", "жк")
    text = re.sub(r"\(([^)]*)\)", " ", text)
    text = text.replace('"', " ")
    text = text.replace("'", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^0-9a-zа-яіїєґ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    parts = []
    for token in text.split():
        clean = token.strip(".")
        parts.append(ABBREVIATIONS.get(clean, clean))
    normalized = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return PHRASE_NORMALIZATIONS.get(normalized, normalized)


def tokenize(name: str) -> List[str]:
    return [token for token in normalize_name(name).split() if token]


def char_ngrams(text: str, size: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return set()
    padded = f" {compact} "
    if len(padded) <= size:
        return {padded}
    return {padded[idx : idx + size] for idx in range(len(padded) - size + 1)}


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def soft_token_similarity(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0

    left_best = [
        max(difflib.SequenceMatcher(None, left_token, right_token).ratio() for right_token in right_tokens)
        for left_token in left_tokens
    ]
    right_best = [
        max(difflib.SequenceMatcher(None, right_token, left_token).ratio() for left_token in left_tokens)
        for right_token in right_tokens
    ]
    return (sum(left_best) + sum(right_best)) / (len(left_best) + len(right_best))


def similarity_components(left: str, right: str) -> Tuple[float, float, float, float]:
    normalized_left = normalize_name(left)
    normalized_right = normalize_name(right)
    if not normalized_left or not normalized_right:
        return 0.0, 0.0, 0.0, 0.0

    seq_score = difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()
    ngram_score = jaccard_similarity(char_ngrams(normalized_left), char_ngrams(normalized_right))
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    exact_token_score = jaccard_similarity(left_tokens, right_tokens)
    soft_token_score = soft_token_similarity(left_tokens, right_tokens)
    token_score = max(exact_token_score, soft_token_score)
    return seq_score, ngram_score, exact_token_score, token_score


def similarity(left: str, right: str) -> float:
    seq_score, ngram_score, _, token_score = similarity_components(left, right)
    if not seq_score and not ngram_score and not token_score:
        return 0.0
    return 0.35 * seq_score + 0.30 * ngram_score + 0.35 * token_score


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_decisions(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        decisions, end = decoder.raw_decode(raw)
        trailing = raw[end:].strip()
        if trailing:
            print(
                f"Попередження: у {path} є зайві дані після JSON; вони будуть проігноровані.",
                file=sys.stderr,
            )
        if not isinstance(decisions, dict):
            raise SystemExit(f"Очікував JSON-об'єкт у {path}, отримав {type(decisions).__name__}") from exc
        return decisions


def save_decisions(path: Path, decisions: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(decisions, handle, ensure_ascii=False, indent=2)


def merge_decisions(
    base_path: Path,
    incoming_path: Path,
) -> Tuple[Dict[str, Dict[str, object]], int, int, List[str]]:
    base_decisions = load_decisions(base_path)
    incoming_decisions = load_decisions(incoming_path)

    added_count = 0
    kept_count = 0
    conflicts: List[str] = []

    for decision_key, incoming_decision in incoming_decisions.items():
        existing_decision = base_decisions.get(decision_key)
        if existing_decision is None:
            base_decisions[decision_key] = incoming_decision
            added_count += 1
            continue

        existing_status = str(existing_decision.get("status", ""))
        incoming_status = str(incoming_decision.get("status", ""))
        if existing_status == incoming_status:
            kept_count += 1
            continue

        conflicts.append(
            f"{decision_key}: основний статус = {existing_status!r}, новий статус = {incoming_status!r}"
        )

    return base_decisions, added_count, kept_count, conflicts


def parse_point_wkt(geometry: str) -> Optional[Tuple[float, float]]:
    match = re.match(r"POINT\s*\(\s*([0-9\.\-]+)\s+([0-9\.\-]+)\s*\)", geometry.strip(), re.IGNORECASE)
    if not match:
        return None
    lon = float(match.group(1))
    lat = float(match.group(2))
    return lon, lat


def osm_link(geometry: str) -> str:
    coords = parse_point_wkt(geometry)
    if coords is None:
        return ""
    lon, lat = coords
    return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=18/{lat:.6f}/{lon:.6f}"


def write_preview_map(output_path: Path, stop_name: str, geometry: str) -> Optional[Path]:
    coords = parse_point_wkt(geometry)
    if coords is None:
        return None
    lon, lat = coords
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>Stop Preview</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .info {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1000;
      max-width: 420px;
      background: rgba(255, 255, 255, 0.94);
      padding: 10px 12px;
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .title {{ font-weight: 700; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <div class="info">
    <div class="title">{stop_name}</div>
    <div>{geometry}</div>
  </div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([{lat}, {lon}], 16);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    L.marker([{lat}, {lon}]).addTo(map).bindPopup({json.dumps(stop_name)}).openPopup();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


class PyppeteerBrowserManager:
    def __init__(self, executable_path: Optional[str] = None) -> None:
        self.executable_path = executable_path or None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.preview_browser = None
        self.eway_browser = None
        self.preview_page = None
        self.eway_page = None
        self.started = False

    def _start_loop(self) -> None:
        if self.loop is not None:
            return

        loop = asyncio.new_event_loop()

        def exception_handler(_loop: asyncio.AbstractEventLoop, context: Dict[str, object]) -> None:
            message = str(context.get("message", ""))
            exception = context.get("exception")
            exception_text = str(exception) if exception is not None else ""
            combined = f"{message} {exception_text}"
            if (
                "Target.detachFromTarget" in combined
                or "No session with given id" in combined
                or "Future exception was never retrieved" in combined
            ):
                return
            _loop.default_exception_handler(context)

        loop.set_exception_handler(exception_handler)

        def runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=runner, name="pyppeteer-browser-loop", daemon=True)
        thread.start()
        self.loop = loop
        self.thread = thread

    def _run_coro(self, coro: "asyncio.Future[bool]") -> bool:
        if self.loop is None:
            return False
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return bool(future.result())

    async def _launch_browser(self):
        if launch is None:
            return None
        launch_kwargs = {
            "headless": False,
            "autoClose": False,
            "handleSIGINT": False,
            "handleSIGTERM": False,
            "handleSIGHUP": False,
            "defaultViewport": None,
            "args": ["--start-maximized"],
        }
        if self.executable_path:
            launch_kwargs["executablePath"] = self.executable_path
        return await launch(launch_kwargs)

    async def _async_goto(self, page, url: str) -> None:
        await page.goto(url, {"waitUntil": "domcontentloaded", "timeout": 30000})
        await page.bringToFront()

    async def _async_ensure_page(self, browser, page):
        if browser is None:
            return None
        if page is None:
            return await browser.newPage()
        try:
            if page.isClosed():
                return await browser.newPage()
        except Exception:
            return await browser.newPage()
        return page

    async def _async_safe_goto(self, browser_attr: str, page_attr: str, url: str) -> bool:
        if not url or not await self._async_start():
            return False
        browser = getattr(self, browser_attr)
        if browser is None:
            return False

        page = await self._async_ensure_page(browser, getattr(self, page_attr))
        if page is None:
            return False
        setattr(self, page_attr, page)

        try:
            await self._async_goto(page, url)
            return True
        except Exception:
            page = await browser.newPage()
            setattr(self, page_attr, page)
            await self._async_goto(page, url)
            return True

    async def _async_start(self) -> bool:
        if self.started:
            return True
        self.preview_browser = await self._launch_browser()
        self.eway_browser = await self._launch_browser()
        if self.preview_browser is None or self.eway_browser is None:
            return False
        self.preview_page = await self.preview_browser.newPage()
        self.eway_page = await self.eway_browser.newPage()
        self.started = True
        return True

    def _ensure_started(self) -> bool:
        if self.started:
            return True
        if launch is None:
            return False
        self._start_loop()
        return self._run_coro(self._async_start())

    async def _async_open_preview(self, url: str) -> bool:
        return await self._async_safe_goto("preview_browser", "preview_page", url)

    def open_preview(self, url: str) -> bool:
        if not self._ensure_started():
            return False
        return self._run_coro(self._async_open_preview(url))

    async def _async_open_eway(self, url: str) -> bool:
        return await self._async_safe_goto("eway_browser", "eway_page", url)

    def open_eway(self, url: str) -> bool:
        if not self._ensure_started():
            return False
        return self._run_coro(self._async_open_eway(url))

    async def _async_open_eway_links(self, urls: Sequence[str]) -> bool:
        clean_urls = [url for url in urls if url]
        if not clean_urls:
            return False
        return await self._async_safe_goto("eway_browser", "eway_page", clean_urls[0])

    def open_eway_links(self, urls: Sequence[str]) -> bool:
        if not self._ensure_started():
            return False
        return self._run_coro(self._async_open_eway_links(urls))

    async def _async_close(self) -> bool:
        for browser in (self.preview_browser, self.eway_browser):
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass
        self.preview_browser = None
        self.eway_browser = None
        self.preview_page = None
        self.eway_page = None
        self.started = False
        return True

    def close(self) -> None:
        if self.loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._async_close(), self.loop).result(timeout=10)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.loop = None
        self.thread = None


PYPPETEER_BROWSER_MANAGER = PyppeteerBrowserManager()
atexit.register(PYPPETEER_BROWSER_MANAGER.close)


def open_stop_map(preview_path: Optional[Path], geometry: str, browser_app: Optional[str] = None) -> None:
    target = preview_path.resolve().as_uri() if preview_path is not None else osm_link(geometry)
    if not target:
        return
    open_url_in_browser(target, browser_app, target_kind="preview")


def open_url_in_browser(url: str, browser_app: Optional[str], target_kind: str = "preview") -> None:
    if not url:
        return
    if browser_app != "system":
        try:
            if target_kind == "eway":
                if PYPPETEER_BROWSER_MANAGER.open_eway(url):
                    return
            else:
                if PYPPETEER_BROWSER_MANAGER.open_preview(url):
                    return
        except Exception as exc:
            print(f"Попередження: не вдалося відкрити URL через pyppeteer: {exc}", file=sys.stderr)
    try:
        webbrowser.open(url, new=0)
    except Exception as exc:
        print(f"Попередження: не вдалося відкрити URL: {exc}", file=sys.stderr)


def build_osm_stops(osm_rows: Sequence[Dict[str, str]]) -> List[OsmStop]:
    stops = []
    for osm_row_id, row in enumerate(osm_rows):
        name = row["name"].strip()
        geometry = row["geometry"].strip()
        stops.append(
            OsmStop(
                osm_row_id=osm_row_id,
                name=name,
                geometry=geometry,
                normalized=normalize_name(name),
            )
        )
    return stops


def build_route_contexts(route_rows: Sequence[Dict[str, str]], route_stop_col: str) -> Dict[int, RouteContext]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for route_row_id, row in enumerate(route_rows):
        key = (
            row.get("calendar", ""),
            row.get("route_id", ""),
            row.get("route", ""),
            row.get("direction", ""),
            row.get("transport", ""),
        )
        grouped[key].append((route_row_id, row))

    contexts: Dict[int, RouteContext] = {}
    for _, pairs in grouped.items():
        pairs_sorted = sorted(
            pairs,
            key=lambda pair: (
                int(pair[1]["index"]) if pair[1].get("index", "").isdigit() else 10**9,
                pair[1].get(route_stop_col, ""),
            ),
        )
        for idx, (route_row_id, row) in enumerate(pairs_sorted):
            prev_stop_name = pairs_sorted[idx - 1][1][route_stop_col].strip() if idx > 0 else ""
            next_stop_name = pairs_sorted[idx + 1][1][route_stop_col].strip() if idx + 1 < len(pairs_sorted) else ""
            contexts[route_row_id] = RouteContext(
                route_row_id=route_row_id,
                stop_id=row.get("stop_id", "").strip(),
                stop_name=row[route_stop_col].strip(),
                route_name=row.get("route", "").strip(),
                route_id=row.get("route_id", "").strip(),
                direction=row.get("direction", "").strip(),
                transport=row.get("transport", "").strip(),
                prev_stop_name=prev_stop_name,
                next_stop_name=next_stop_name,
            )
    return contexts


def format_route_context(context: RouteContext) -> str:
    prev_label = context.prev_stop_name or "START"
    next_label = context.next_stop_name or "END"
    return (
        f'маршрут {context.route_name} ({context.transport}, {context.direction}) | '
        f'попередня: {prev_label} | поточна: {context.stop_name} | наступна: {next_label}'
    )


def summarize_candidate_contexts(contexts: Sequence[RouteContext]) -> str:
    return f"маршрутів: {len(group_context_options(contexts))}"


def group_context_options(contexts: Sequence[RouteContext]) -> List[ContextOption]:
    grouped: Dict[Tuple[str, str, str, str, str], List[RouteContext]] = defaultdict(list)
    for context in contexts:
        key = (
            context.route_name,
            context.route_id,
            context.direction,
            context.transport,
            format_route_context(context),
        )
        grouped[key].append(context)

    options: List[ContextOption] = []
    for grouped_contexts in grouped.values():
        route_row_ids = tuple(sorted(context.route_row_id for context in grouped_contexts))
        representative = min(grouped_contexts, key=lambda context: context.route_row_id)
        options.append(ContextOption(route_row_ids=route_row_ids, context=representative))

    options.sort(
        key=lambda option: (
            stop_id_sort_key(option.context.stop_id),
            option.context.route_name,
            option.context.transport,
            direction_sort_key(option.context.direction),
            option.route_row_ids[0],
        )
    )
    return options


def context_group_key(option: ContextOption) -> Tuple[str, str]:
    return option.context.route_name, option.context.transport


def context_group_label(option: ContextOption) -> str:
    return f"маршрут {option.context.route_name} ({option.context.transport})"


def stop_id_group_key(option: ContextOption) -> str:
    return option.context.stop_id or ""


def stop_id_group_label(stop_id: str) -> str:
    return f"stop_id: {stop_id}" if stop_id else "stop_id: відсутній"


def stop_id_group_border(stop_id: str) -> str:
    label = f"#### {stop_id_group_label(stop_id)} ####"
    return "#" * len(label)


def stop_id_link(stop_id: str) -> str:
    normalized = stop_id.strip()
    if not normalized:
        return ""
    return f"https://www.eway.in.ua/ua/cities/kyiv/stops/{normalized}"


def build_eway_links_for_options(options: Sequence[ContextOption]) -> List[str]:
    links: List[str] = []
    for option in options:
        link = stop_id_link(option.context.stop_id)
        if link and link not in links:
            links.append(link)
    return links


def direction_sort_key(direction: str) -> Tuple[int, str]:
    normalized = direction.strip().lower()
    if normalized == "forward":
        return 0, normalized
    if normalized == "backward":
        return 1, normalized
    return 2, normalized


def stop_id_sort_key(stop_id: str) -> Tuple[int, int, str]:
    normalized = stop_id.strip()
    if not normalized:
        return 2, 0, ""
    if normalized.isdigit():
        return 0, int(normalized), normalized
    return 1, 0, normalized


def build_name_to_unassigned_contexts(
    route_rows: Sequence[Dict[str, str]],
    route_contexts: Dict[int, RouteContext],
    assigned_route_row_ids: Iterable[int],
    route_stop_col: str,
) -> Dict[str, List[RouteContext]]:
    assigned = set(assigned_route_row_ids)
    by_name: Dict[str, List[RouteContext]] = defaultdict(list)
    for route_row_id, row in enumerate(route_rows):
        if route_row_id in assigned:
            continue
        stop_name = row[route_stop_col].strip()
        by_name[stop_name].append(route_contexts[route_row_id])
    return by_name


def generate_candidate_names(
    osm_stop_name: str,
    available_names: Iterable[str],
    limit: int,
) -> List[Tuple[float, str]]:
    normalized_target = normalize_name(osm_stop_name)
    candidates: List[Tuple[float, str]] = []

    for route_name in available_names:
        normalized_name = normalize_name(route_name)
        if not normalized_name:
            continue
        if normalized_name == normalized_target:
            candidates.append((1.0, route_name))
            continue
        seq_score, ngram_score, exact_token_score, token_score = similarity_components(osm_stop_name, route_name)
        score = 0.35 * seq_score + 0.30 * ngram_score + 0.35 * token_score
        if score < 0.5:
            continue
        if max(ngram_score, token_score) < 0.3 and seq_score < 0.75:
            continue
        if exact_token_score == 0.0 and token_score < 0.45 and ngram_score < 0.35:
            continue
        candidates.append((score, route_name))

    best_by_name: Dict[str, float] = {}
    for score, name in candidates:
        if score > best_by_name.get(name, -1.0):
            best_by_name[name] = score
    ranked = sorted(((score, name) for name, score in best_by_name.items()), key=lambda item: (-item[0], item[1]))
    return ranked[:limit]


def ask_choice(
    stop: OsmStop,
    search_name: str,
    candidate_names: Sequence[Tuple[float, str]],
    context_by_name: Dict[str, List[RouteContext]],
) -> Optional[str]:
    print()
    print(f'Osm stop: "{stop.name}"')
    print(f"  geometry: {stop.geometry}")
    print(f"  osm map: {osm_link(stop.geometry)}")
    search_name_label = format_search_name_label(stop, search_name)
    if search_name_label is not None:
        print(f"  {search_name_label}")
    if not candidate_names:
        print("  Схожих назв не знайдено.")
        return None
    for idx, (score, route_name) in enumerate(candidate_names, start=1):
        summary = summarize_candidate_contexts(context_by_name.get(route_name, []))
        print(f"  {idx}. [{score:.3f}] {route_name} | {summary}")
    print("  n. немає підходящої назви")
    while True:
        raw = input("Вибери номер назви, Enter = 1, n = немає: ").strip().lower()
        if raw == "":
            return candidate_names[0][1]
        if raw == "n":
            return None
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(candidate_names):
                return candidate_names[index][1]
        print("Некоректний ввід.")


def auto_select_exact_candidate(candidate_names: Sequence[Tuple[float, str]]) -> Optional[str]:
    exact_matches = [name for score, name in candidate_names if score == 1.0]
    if len(exact_matches) == 1:
        return exact_matches[0]
    return None


def _wrap_lines(text: str, width: int) -> List[str]:
    safe_width = max(20, width)
    wrapped = textwrap.wrap(text, width=safe_width, replace_whitespace=False, drop_whitespace=False)
    return wrapped or [""]


def _draw_block(stdscr: "curses._CursesWindow", y: int, width: int, text: str, attr: int = 0) -> int:
    max_y, _ = stdscr.getmaxyx()
    for line in _wrap_lines(text, width):
        if y >= max_y - 1:
            return y
        stdscr.addnstr(y, 0, line, width, attr)
        y += 1
    return y


def format_search_name_label(stop: OsmStop, search_name: str) -> Optional[str]:
    normalized_search = search_name.strip()
    if not normalized_search or normalized_search == stop.name:
        return None
    return f'Пошук за назвою: "{normalized_search}" (замість OSM "{stop.name}")'


def prompt_search_name_tui(stdscr: "curses._CursesWindow", initial_value: str) -> Optional[str]:
    curses.echo()
    curses.curs_set(1)
    max_y, max_x = stdscr.getmaxyx()
    prompt = "Нова назва для пошуку: "
    stdscr.move(max_y - 1, 0)
    stdscr.clrtoeol()
    stdscr.addnstr(max_y - 1, 0, prompt, max_x - 1, curses.A_BOLD)
    stdscr.addnstr(max_y - 1, len(prompt), initial_value, max_x - len(prompt) - 1)
    stdscr.refresh()
    try:
        raw = stdscr.getstr(max_y - 1, len(prompt), max(1, max_x - len(prompt) - 2))
    except curses.error:
        raw = b""
    finally:
        curses.noecho()
        curses.curs_set(0)
    value = raw.decode("utf-8", errors="ignore").strip()
    return value or None


def build_context_display_rows(
    candidate_options: Sequence[ContextOption],
    selected_rows: set[int],
    selected_stop_group: str,
) -> Tuple[List[DisplayRow], int]:
    rows: List[DisplayRow] = []
    selected_display_row = 0
    previous_stop_id_group: Optional[str] = None

    for option_index, option in enumerate(candidate_options):
        stop_id_group = stop_id_group_key(option)
        group_attr = curses.A_REVERSE if stop_id_group == selected_stop_group else 0
        if stop_id_group != previous_stop_id_group:
            stop_id_route_row_ids = {
                route_row_id
                for grouped_option in candidate_options
                if stop_id_group_key(grouped_option) == stop_id_group
                for route_row_id in grouped_option.route_row_ids
            }
            checked = "[x]" if stop_id_route_row_ids and all(route_row_id in selected_rows for route_row_id in stop_id_route_row_ids) else "[ ]"
            prefix = ">" if stop_id_group == selected_stop_group else " "
            rows.append(
                DisplayRow(
                    text=f"{prefix} #### {checked} {stop_id_group_label(stop_id_group)} ####",
                    attr=group_attr,
                    option_index=option_index,
                )
            )
            if stop_id_group == selected_stop_group:
                selected_display_row = len(rows) - 1
            previous_stop_id_group = stop_id_group

    return rows, selected_display_row


def build_stop_group_keys(candidate_options: Sequence[ContextOption]) -> List[str]:
    keys: List[str] = []
    for option in candidate_options:
        stop_id_group = stop_id_group_key(option)
        if stop_id_group not in keys:
            keys.append(stop_id_group)
    return keys


def rebuild_route_assignments(
    decisions: Dict[str, Dict[str, object]],
) -> Dict[int, Tuple[str, str]]:
    route_assignments: Dict[int, Tuple[str, str]] = {}
    apply_saved_decisions(decisions, route_assignments)
    return route_assignments


def find_previous_decision_key(
    osm_stops: Sequence[OsmStop],
    decisions: Dict[str, Dict[str, object]],
    current_index: int,
    start_osm_row_id: int,
) -> Optional[str]:
    for index in range(current_index - 1, -1, -1):
        stop = osm_stops[index]
        if stop.osm_row_id < start_osm_row_id:
            continue
        decision_key = str(stop.osm_row_id)
        if decision_key in decisions:
            return decision_key
    return None


def build_osm_no_candidates_rows(
    osm_stops: Sequence[OsmStop],
    decisions: Dict[str, Dict[str, object]],
) -> List[Dict[str, str]]:
    by_key = {str(stop.osm_row_id): stop for stop in osm_stops}
    rows: List[Dict[str, str]] = []
    for decision_key, decision in decisions.items():
        if decision.get("status") != "no_candidates":
            continue
        stop = by_key.get(decision_key)
        if stop is None:
            continue
        rows.append({"osm_row_id": decision_key, "name": stop.name, "geometry": stop.geometry})
    rows.sort(key=lambda row: int(row["osm_row_id"]))
    return rows


def flatten_display_rows(display_rows: Sequence[DisplayRow], width: int) -> List[Tuple[str, int]]:
    flat_rows: List[Tuple[str, int]] = []
    for row in display_rows:
        for line in _wrap_lines(row.text, width):
            flat_rows.append((line, row.attr))
    return flat_rows


def _pick_name_and_contexts_tui(
    stop: OsmStop,
    search_name: str,
    candidate_names: Sequence[Tuple[float, str]],
    context_by_name: Dict[str, List[RouteContext]],
    preview_map_path: Optional[Path],
    auto_select_exact: bool,
    progress_label: str,
    open_browser: bool,
    preview_browser_app: Optional[str],
    eway_browser_app: Optional[str],
    initial_selected_name: Optional[str] = None,
    initial_selected_route_row_ids: Optional[Sequence[int]] = None,
) -> InteractionResult:
    preview_path = write_preview_map(preview_map_path, stop.name, stop.geometry) if preview_map_path else None
    if open_browser:
        open_stop_map(preview_path, stop.geometry, preview_browser_app)
    initial_name = None
    if initial_selected_name:
        initial_name = initial_selected_name
    elif auto_select_exact:
        initial_name = auto_select_exact_candidate(candidate_names)

    def run(stdscr: "curses._CursesWindow") -> InteractionResult:
        curses.curs_set(0)
        stdscr.keypad(True)

        stage = "contexts" if initial_name is not None and candidate_names else "names"
        name_index = 0
        if initial_name is not None:
            for idx, (_, route_name) in enumerate(candidate_names):
                if route_name == initial_name:
                    name_index = idx
                    break

        context_index = 0
        selected_rows: set[int] = set(int(route_row_id) for route_row_id in (initial_selected_route_row_ids or []))
        context_position_initialized = False
        message = ""
        last_opened_eway_links: Optional[Tuple[str, ...]] = None

        while True:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            width = max_x - 1
            y = 0

            y = _draw_block(stdscr, y, width, progress_label, curses.A_BOLD)
            y = _draw_block(stdscr, y, width, f'OSM stop: "{stop.name}"', curses.A_BOLD)
            y = _draw_block(stdscr, y, width, f"geometry: {stop.geometry}")
            y = _draw_block(stdscr, y, width, f"osm map: {osm_link(stop.geometry)}")
            search_name_label = format_search_name_label(stop, search_name)
            if search_name_label is not None:
                y = _draw_block(stdscr, y, width, search_name_label, curses.A_BOLD)
            if preview_path is not None:
                y = _draw_block(stdscr, y, width, f"preview file: {preview_path}")
            y += 1

            if stage == "names":
                if candidate_names:
                    stage_prompt = "Етап 1/2. Стрілки: вибір назви. Enter: далі. r: змінити назву пошуку. s: пропустити. n: немає підходящої назви. b: назад до попередньої зупинки. q: вийти."
                else:
                    stage_prompt = "Етап 1/2. Кандидатів не знайдено. r: змінити назву пошуку. s: пропустити. n: підтвердити, що кандидатів немає. b: назад до попередньої зупинки. q: вийти."
                y = _draw_block(
                    stdscr,
                    y,
                    width,
                    stage_prompt,
                    curses.A_BOLD,
                )
                if not candidate_names:
                    y = _draw_block(stdscr, y, width, "Схожих назв не знайдено.", curses.A_BOLD)
                else:
                    visible_rows = max(1, max_y - y - 2)
                    start = 0
                    if name_index >= visible_rows:
                        start = name_index - visible_rows + 1
                    current = start
                    while current < len(candidate_names) and y < max_y - 1:
                        score, route_name = candidate_names[current]
                        summary = summarize_candidate_contexts(context_by_name.get(route_name, []))
                        prefix = ">" if current == name_index else " "
                        attr = curses.A_REVERSE if current == name_index else 0
                        y = _draw_block(stdscr, y, width, f"{prefix} [{score:.3f}] {route_name} | {summary}", attr)
                        current += 1
                if message and y < max_y - 1:
                    stdscr.addnstr(max_y - 1, 0, message, width, curses.A_BOLD)

                key = stdscr.getch()
                if candidate_names and key in (curses.KEY_UP, ord("k")):
                    name_index = (name_index - 1) % len(candidate_names)
                    message = ""
                elif candidate_names and key in (curses.KEY_DOWN, ord("j")):
                    name_index = (name_index + 1) % len(candidate_names)
                    message = ""
                elif candidate_names and key in (10, 13, curses.KEY_ENTER):
                    selected_name = candidate_names[name_index][1]
                    candidate_contexts = context_by_name.get(selected_name, [])
                    if not candidate_contexts:
                        message = "Для цієї назви вже не лишилось рядків без geometry."
                        continue
                    context_index = 0
                    selected_rows = set()
                    stage = "contexts"
                    message = ""
                elif key in (ord("r"), ord("R")):
                    renamed_search = prompt_search_name_tui(stdscr, search_name)
                    if renamed_search:
                        return InteractionResult(status="rename", search_name=renamed_search)
                    message = "Назву пошуку не змінено."
                elif key in (ord("s"), ord("S")):
                    return InteractionResult(status="skipped", search_name=search_name)
                elif key in (ord("n"), ord("N")):
                    return InteractionResult(status="no_candidates", search_name=search_name)
                elif key in (ord("b"), ord("B"), ord("p"), ord("P")):
                    return InteractionResult(status="back", search_name=search_name)
                elif key in (ord("q"), ord("Q")):
                    return InteractionResult(status="quit", search_name=search_name)
            else:
                selected_name = candidate_names[name_index][1]
                candidate_contexts = context_by_name.get(selected_name, [])
                candidate_options = group_context_options(candidate_contexts)
                if not candidate_options:
                    stage = "names"
                    message = "Для цієї назви вже не лишилось рядків без geometry."
                    continue
                stop_group_keys = build_stop_group_keys(candidate_options)
                if not stop_group_keys:
                    stage = "names"
                    message = "Для цієї назви вже не лишилось рядків без geometry."
                    continue
                if context_index >= len(stop_group_keys):
                    context_index = len(stop_group_keys) - 1
                if not context_position_initialized and selected_rows:
                    for idx, stop_group_key in enumerate(stop_group_keys):
                        stop_group_route_row_ids = {
                            route_row_id
                            for option in candidate_options
                            if stop_id_group_key(option) == stop_group_key
                            for route_row_id in option.route_row_ids
                        }
                        if any(route_row_id in selected_rows for route_row_id in stop_group_route_row_ids):
                            context_index = idx
                            break
                    context_position_initialized = True
                current_stop_group = stop_group_keys[context_index]
                eway_links = tuple(build_eway_links_for_options(candidate_options))
                if eway_browser_app != "system" and eway_links != last_opened_eway_links:
                    try:
                        PYPPETEER_BROWSER_MANAGER.open_eway_links(eway_links)
                    except Exception as exc:
                        message = f"Не вдалося відкрити easyway-посилання: {exc}"
                    last_opened_eway_links = eway_links

                auto_note = (
                    " Точний збіг вибрано автоматично; Esc повертає до списку назв."
                    if initial_name == selected_name and auto_select_exact
                    else ""
                )
                y = _draw_block(
                    stdscr,
                    y,
                    width,
                    f'Етап 2/2. Назва: "{selected_name}". Стрілки: рух. Space: вибір. Enter: підтвердити. Esc: назад. b: попередня зупинка.{auto_note}',
                    curses.A_BOLD,
                )
                available_lines = max(1, max_y - y - 2)
                display_rows, selected_display_row = build_context_display_rows(
                    candidate_options=candidate_options,
                    selected_rows=selected_rows,
                    selected_stop_group=current_stop_group,
                )
                flat_rows = flatten_display_rows(display_rows, width)
                anchor_row = 0
                consumed = 0
                for display_row in display_rows[:selected_display_row]:
                    consumed += len(_wrap_lines(display_row.text, width))
                anchor_row = consumed
                start_row = max(0, anchor_row - available_lines + 1)
                end_row = min(len(flat_rows), start_row + available_lines)
                while start_row > 0 and end_row - start_row < available_lines:
                    start_row -= 1
                for line, attr in flat_rows[start_row:end_row]:
                    if y >= max_y - 2:
                        break
                    if y >= max_y - 1:
                        break
                    stdscr.addnstr(y, 0, line, width, attr)
                    y += 1
                footer = "Стрілки: stop_id, Space/s: toggle stop_id, a/Ctrl+A: усі, e: eway, c: очистити, q: вийти"
                if y < max_y - 1:
                    stdscr.addnstr(max_y - 2, 0, footer, width, curses.A_DIM)
                if message:
                    stdscr.addnstr(max_y - 1, 0, message, width, curses.A_BOLD)

                key = stdscr.getch()
                if key in (curses.KEY_UP, ord("k")):
                    context_index = (context_index - 1) % len(stop_group_keys)
                    message = ""
                elif key in (curses.KEY_DOWN, ord("j")):
                    context_index = (context_index + 1) % len(stop_group_keys)
                    message = ""
                elif key == 27:
                    stage = "names"
                    message = ""
                elif key in (ord("b"), ord("B"), ord("p"), ord("P")):
                    return InteractionResult(status="back", search_name=search_name)
                elif key in (1, ord("a"), ord("A")):
                    selected_rows = {
                        route_row_id for option in candidate_options for route_row_id in option.route_row_ids
                    }
                    message = ""
                elif key in (ord(" "), ord("s"), ord("S")):
                    current_stop_id = current_stop_group
                    stop_id_route_row_ids = {
                        route_row_id
                        for option in candidate_options
                        if option.context.stop_id == current_stop_id
                        for route_row_id in option.route_row_ids
                    }
                    if stop_id_route_row_ids and all(route_row_id in selected_rows for route_row_id in stop_id_route_row_ids):
                        selected_rows.difference_update(stop_id_route_row_ids)
                    else:
                        selected_rows.update(stop_id_route_row_ids)
                    message = ""
                elif key in (ord("e"), ord("E")):
                    current_stop_id = current_stop_group
                    link = stop_id_link(current_stop_id)
                    if link:
                        open_url_in_browser(link, eway_browser_app, target_kind="eway")
                        message = f"Відкрито eway для stop_id {current_stop_id}"
                    else:
                        message = "Для цього елемента немає stop_id."
                elif key in (ord("c"), ord("C")):
                    selected_rows.clear()
                    message = ""
                elif key in (ord("q"), ord("Q")):
                    return InteractionResult(status="quit", search_name=search_name)
                elif key in (10, 13, curses.KEY_ENTER):
                    return InteractionResult(
                        status="selected",
                        search_name=search_name,
                        selected_name=selected_name,
                        selected_route_row_ids=[
                            route_row_id
                            for option in candidate_options
                            for route_row_id in option.route_row_ids
                            if route_row_id in selected_rows
                        ],
                    )

        return InteractionResult(status="no_candidates", search_name=search_name)

    return curses.wrapper(run)


def parse_multi_indices(raw: str, limit: int) -> Optional[List[int]]:
    values = raw.split()
    if not values:
        return []
    result: List[int] = []
    for value in values:
        if not value.isdigit():
            return None
        index = int(value)
        if index < 1 or index > limit:
            return None
        if index not in result:
            result.append(index)
    return result


def ask_context_rows(
    stop: OsmStop,
    selected_name: str,
    contexts: Sequence[RouteContext],
    preview_map_path: Optional[Path],
    eway_browser_app: Optional[str],
    initial_selected_route_row_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    preview_path = write_preview_map(preview_map_path, stop.name, stop.geometry) if preview_map_path else None
    options = group_context_options(contexts)
    eway_links = build_eway_links_for_options(options)
    if eway_browser_app != "system" and eway_links:
        try:
            PYPPETEER_BROWSER_MANAGER.open_eway_links(eway_links)
        except Exception as exc:
            print(f"Попередження: не вдалося відкрити easyway-посилання: {exc}", file=sys.stderr)

    print()
    print(f'Вибрана назва з kyiv-routes: "{selected_name}"')
    print(f'OSM stop: "{stop.name}"')
    print(f"  geometry: {stop.geometry}")
    print(f"  osm map: {osm_link(stop.geometry)}")
    if preview_path is not None:
        print(f"  preview file: {preview_path}")
    stop_group_keys = build_stop_group_keys(options)
    selected_rows = set(int(route_row_id) for route_row_id in (initial_selected_route_row_ids or []))
    previous_stop_id_group: Optional[str] = None
    group_display_index = 0
    for option in options:
        stop_id_group = stop_id_group_key(option)
        if stop_id_group != previous_stop_id_group:
            group_display_index += 1
            if previous_stop_id_group is not None:
                print()
            stop_id_route_row_ids = [
                route_row_id
                for grouped_option in options
                if stop_id_group_key(grouped_option) == stop_id_group
                for route_row_id in grouped_option.route_row_ids
            ]
            marker = "[x]" if stop_id_route_row_ids and all(route_row_id in selected_rows for route_row_id in stop_id_route_row_ids) else "[ ]"
            print(f"  {group_display_index}. #### {marker} {stop_id_group_label(stop_id_group)} ####")
            previous_stop_id_group = stop_id_group
    if selected_rows:
        print("Enter = залишити попередній вибір.")
    print("Введи кілька номерів через пробіл, a/all = усі, sid:<id> = весь stop_id, e:<id> = відкрити eway, Enter = жодного/залишити вибір, b = назад до назв, p = попередня зупинка")

    while True:
        raw = input("Контексти: ").strip().lower()
        if raw == "":
            return list(initial_selected_route_row_ids or [])
        if raw in {"a", "all"}:
            return [route_row_id for option in options for route_row_id in option.route_row_ids]
        if raw == "b":
            return [-1]
        if raw in {"p", "prev", "back"}:
            return [-2]
        if raw.startswith("sid:"):
            stop_id = raw[4:].strip()
            selected_route_row_ids = [
                route_row_id
                for option in options
                if option.context.stop_id == stop_id
                for route_row_id in option.route_row_ids
            ]
            if selected_route_row_ids:
                return selected_route_row_ids
            print("Некоректний stop_id.")
            continue
        if raw.startswith("e:"):
            stop_id = raw[2:].strip()
            link = stop_id_link(stop_id)
            if link:
                open_url_in_browser(link, eway_browser_app, target_kind="eway")
                print(f"Відкрито eway для stop_id {stop_id}")
            else:
                print("Некоректний stop_id.")
            continue
        parsed = parse_multi_indices(raw, len(stop_group_keys))
        if parsed is not None:
            return [
                route_row_id
                for index in parsed
                for option in options
                if stop_id_group_key(option) == stop_group_keys[index - 1]
                for route_row_id in option.route_row_ids
            ]
        print("Некоректний ввід.")


def pick_name_and_contexts(
    stop: OsmStop,
    context_by_name: Dict[str, List[RouteContext]],
    candidate_limit: int,
    preview_map_path: Optional[Path],
    auto_select_exact: bool,
    progress_label: str,
    open_browser: bool,
    preview_browser_app: Optional[str],
    eway_browser_app: Optional[str],
    initial_search_name: Optional[str] = None,
    initial_selected_name: Optional[str] = None,
    initial_selected_route_row_ids: Optional[Sequence[int]] = None,
) -> InteractionResult:
    search_name = initial_search_name or stop.name
    pending_initial_selected_name = initial_selected_name
    if sys.stdin.isatty() and sys.stdout.isatty():
        while True:
            candidate_names = generate_candidate_names(search_name, context_by_name.keys(), candidate_limit)
            try:
                result = _pick_name_and_contexts_tui(
                    stop=stop,
                    search_name=search_name,
                    candidate_names=candidate_names,
                    context_by_name=context_by_name,
                    preview_map_path=preview_map_path,
                    auto_select_exact=auto_select_exact,
                    progress_label=progress_label,
                    open_browser=open_browser,
                    preview_browser_app=preview_browser_app,
                    eway_browser_app=eway_browser_app,
                    initial_selected_name=initial_selected_name,
                    initial_selected_route_row_ids=initial_selected_route_row_ids,
                )
            except curses.error:
                break
            if result.status == "rename":
                search_name = result.search_name
                continue
            return result

    preview_path = write_preview_map(preview_map_path, stop.name, stop.geometry) if preview_map_path else None
    if open_browser:
        open_stop_map(preview_path, stop.geometry, preview_browser_app)

    while True:
        candidate_names = generate_candidate_names(search_name, context_by_name.keys(), candidate_limit)
        selected_name = None
        reused_previous_selection = False
        if pending_initial_selected_name and any(route_name == pending_initial_selected_name for _, route_name in candidate_names):
            selected_name = pending_initial_selected_name
            pending_initial_selected_name = None
            reused_previous_selection = True
            print()
            print(progress_label)
            print(f'Osm stop: "{stop.name}"')
            search_name_label = format_search_name_label(stop, search_name)
            if search_name_label is not None:
                print(search_name_label)
            print(f'Повторно відкрито попередній вибір назви: "{selected_name}"')
        elif auto_select_exact:
            selected_name = auto_select_exact_candidate(candidate_names)
        if selected_name is not None and candidate_names and not reused_previous_selection:
            print()
            print(progress_label)
            print(f'Osm stop: "{stop.name}"')
            search_name_label = format_search_name_label(stop, search_name)
            if search_name_label is not None:
                print(search_name_label)
            print(f'Автоматично вибрано точний збіг назви: "{selected_name}"')
        else:
            print()
            print(progress_label)
            selected_name = ask_choice(stop, search_name, candidate_names, context_by_name)
            if selected_name is None:
                raw = input("r = змінити назву пошуку, s = пропустити, n = немає кандидатів, p = попередня зупинка, q = вийти: ").strip().lower()
                if raw == "r":
                    renamed_search = input("Нова назва для пошуку: ").strip()
                    if renamed_search:
                        search_name = renamed_search
                    continue
                if raw == "s":
                    return InteractionResult(status="skipped", search_name=search_name)
                if raw in {"p", "prev", "back", "b"}:
                    return InteractionResult(status="back", search_name=search_name)
                if raw == "q":
                    return InteractionResult(status="quit", search_name=search_name)
                return InteractionResult(status="no_candidates", search_name=search_name)

        candidate_contexts = context_by_name.get(selected_name, [])
        if not candidate_contexts:
            print("Для цієї назви вже не лишилось рядків без geometry.")
            candidate_names = [(score, name) for score, name in candidate_names if name != selected_name]
            if not candidate_names:
                raw = input("r = змінити назву пошуку, s = пропустити, n = немає кандидатів, p = попередня зупинка, q = вийти: ").strip().lower()
                if raw == "r":
                    renamed_search = input("Нова назва для пошуку: ").strip()
                    if renamed_search:
                        search_name = renamed_search
                    continue
                if raw == "s":
                    return InteractionResult(status="skipped", search_name=search_name)
                if raw in {"p", "prev", "back", "b"}:
                    return InteractionResult(status="back", search_name=search_name)
                if raw == "q":
                    return InteractionResult(status="quit", search_name=search_name)
                return InteractionResult(status="no_candidates", search_name=search_name)
            continue

        initial_rows_for_name = initial_selected_route_row_ids if initial_selected_name == selected_name else None
        selected_route_row_ids = ask_context_rows(
            stop,
            selected_name,
            candidate_contexts,
            None,
            eway_browser_app,
            initial_selected_route_row_ids=initial_rows_for_name,
        )
        if selected_route_row_ids == [-1]:
            continue
        if selected_route_row_ids == [-2]:
            return InteractionResult(status="back", search_name=search_name)
        return InteractionResult(
            status="selected",
            search_name=search_name,
            selected_name=selected_name,
            selected_route_row_ids=selected_route_row_ids,
        )


def apply_saved_decisions(
    decisions: Dict[str, Dict[str, object]],
    route_assignments: Dict[int, Tuple[str, str]],
) -> None:
    for decision in decisions.values():
        if decision.get("status") != "matched":
            continue
        geometry = str(decision.get("geometry", ""))
        osm_name = str(decision.get("osm_name", ""))
        for route_row_id in decision.get("route_row_ids", []):
            route_assignments[int(route_row_id)] = (geometry, osm_name)


def extract_stop_ids(
    route_contexts: Dict[int, RouteContext],
    route_row_ids: Sequence[int],
) -> List[str]:
    stop_ids: List[str] = []
    for route_row_id in route_row_ids:
        context = route_contexts.get(int(route_row_id))
        if context is None or not context.stop_id:
            continue
        if context.stop_id not in stop_ids:
            stop_ids.append(context.stop_id)
    return stop_ids


def build_output_rows(
    route_rows: Sequence[Dict[str, str]],
    route_assignments: Dict[int, Tuple[str, str]],
) -> List[Dict[str, str]]:
    output_rows: List[Dict[str, str]] = []
    for route_row_id, row in enumerate(route_rows):
        new_row = dict(row)
        geometry, osm_name = route_assignments.get(route_row_id, ("", ""))
        new_row["geometry"] = geometry
        new_row["osm_stop_name"] = osm_name
        output_rows.append(new_row)
    return output_rows


def run_matching(
    osm_rows: Sequence[Dict[str, str]],
    route_rows: Sequence[Dict[str, str]],
    route_stop_col: str,
    decisions_path: Path,
    candidate_limit: int,
    preview_map_path: Optional[Path],
    auto_select_exact: bool,
    open_browser: bool,
    preview_browser_app: Optional[str],
    eway_browser_app: Optional[str],
    start_osm_row_id: int,
    review_osm_row_ids: Optional[Set[str]] = None,
    review_mismatches_path: Optional[Path] = None,
    revisit_skipped: bool = False,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], Dict[str, Dict[str, object]]]:
    osm_stops = build_osm_stops(osm_rows)
    route_contexts = build_route_contexts(route_rows, route_stop_col)
    decisions = load_decisions(decisions_path)
    review_mismatches = load_decisions(review_mismatches_path) if review_mismatches_path else {}
    route_assignments = rebuild_route_assignments(decisions)
    revisit_state: Dict[str, Tuple[Optional[str], Optional[str], List[int]]] = {}

    total_stops = len(osm_stops)
    stop_index = 0
    while stop_index < total_stops:
        stop = osm_stops[stop_index]
        if stop.osm_row_id < start_osm_row_id:
            stop_index += 1
            continue
        decision_key = str(stop.osm_row_id)
        if review_osm_row_ids is not None and decision_key not in review_osm_row_ids:
            stop_index += 1
            continue
        progress_label = f"Переглянуто: {stop_index + 1} / {total_stops}"
        existing_decision = decisions.get(decision_key)
        if existing_decision is not None:
            existing_status = str(existing_decision.get("status", ""))
            should_revisit_existing = (revisit_skipped and existing_status == "skipped") or (
                review_osm_row_ids is not None and decision_key in review_osm_row_ids
            )
            if not should_revisit_existing:
                stop_index += 1
                continue
            revisit_state[decision_key] = (
                str(existing_decision.get("search_name") or stop.name),
                str(existing_decision.get("candidate_name")) if existing_decision.get("candidate_name") is not None else None,
                [int(route_row_id) for route_row_id in existing_decision.get("route_row_ids", [])],
            )
            decisions.pop(decision_key, None)

        context_by_name = build_name_to_unassigned_contexts(
            route_rows=route_rows,
            route_contexts=route_contexts,
            assigned_route_row_ids=route_assignments.keys(),
            route_stop_col=route_stop_col,
        )
        if not context_by_name:
            break

        advance_to_next = True
        while True:
            initial_search_name = None
            initial_selected_name = None
            initial_selected_route_row_ids = None
            if decision_key in revisit_state:
                initial_search_name, initial_selected_name, initial_selected_route_row_ids = revisit_state.pop(decision_key)
            interaction = pick_name_and_contexts(
                stop=stop,
                context_by_name=context_by_name,
                candidate_limit=candidate_limit,
                preview_map_path=preview_map_path,
                auto_select_exact=auto_select_exact,
                progress_label=progress_label,
                open_browser=open_browser,
                preview_browser_app=preview_browser_app,
                eway_browser_app=eway_browser_app,
                initial_search_name=initial_search_name,
                initial_selected_name=initial_selected_name,
                initial_selected_route_row_ids=initial_selected_route_row_ids,
            )
            search_name_changed = interaction.search_name != stop.name
            if interaction.status == "quit":
                raise SystemExit(0)
            if interaction.status == "back":
                previous_decision_key = find_previous_decision_key(osm_stops, decisions, stop_index, start_osm_row_id)
                if previous_decision_key is None:
                    continue
                previous_decision = decisions.get(previous_decision_key, {})
                revisit_state[previous_decision_key] = (
                    str(previous_decision.get("search_name") or ""),
                    str(previous_decision.get("candidate_name")) if previous_decision.get("candidate_name") is not None else None,
                    [int(route_row_id) for route_row_id in previous_decision.get("route_row_ids", [])],
                )
                decisions.pop(previous_decision_key, None)
                save_decisions(decisions_path, decisions)
                route_assignments = rebuild_route_assignments(decisions)
                previous_stop_index = next(
                    (index for index, previous_stop in enumerate(osm_stops) if str(previous_stop.osm_row_id) == previous_decision_key),
                    stop_index,
                )
                stop_index = previous_stop_index
                advance_to_next = False
                break
            if interaction.status == "no_candidates":
                decisions[decision_key] = {
                    "status": "no_candidates",
                    "osm_name": stop.name,
                    "geometry": stop.geometry,
                    "search_name": interaction.search_name,
                    "search_name_changed": search_name_changed,
                }
                save_decisions(decisions_path, decisions)
                if review_mismatches_path and decision_key in review_mismatches:
                    review_mismatches.pop(decision_key, None)
                    save_decisions(review_mismatches_path, review_mismatches)
                break

            if interaction.status == "skipped":
                decisions[decision_key] = {
                    "status": "skipped",
                    "osm_name": stop.name,
                    "geometry": stop.geometry,
                    "search_name": interaction.search_name,
                    "search_name_changed": search_name_changed,
                    "candidate_name": None,
                    "route_row_ids": [],
                }
                save_decisions(decisions_path, decisions)
                if review_mismatches_path and decision_key in review_mismatches:
                    review_mismatches.pop(decision_key, None)
                    save_decisions(review_mismatches_path, review_mismatches)
                break

            if not interaction.selected_route_row_ids:
                decisions[decision_key] = {
                    "status": "skipped",
                    "osm_name": stop.name,
                    "geometry": stop.geometry,
                    "search_name": interaction.search_name,
                    "search_name_changed": search_name_changed,
                    "candidate_name": interaction.selected_name,
                    "route_row_ids": [],
                }
                save_decisions(decisions_path, decisions)
                if review_mismatches_path and decision_key in review_mismatches:
                    review_mismatches.pop(decision_key, None)
                    save_decisions(review_mismatches_path, review_mismatches)
                break

            for route_row_id in interaction.selected_route_row_ids:
                route_assignments[route_row_id] = (stop.geometry, stop.name)

            stop_ids = extract_stop_ids(route_contexts, interaction.selected_route_row_ids)
            decisions[decision_key] = {
                "status": "matched",
                "osm_name": stop.name,
                "geometry": stop.geometry,
                "search_name": interaction.search_name,
                "search_name_changed": search_name_changed,
                "candidate_name": interaction.selected_name,
                "stop_id": stop_ids[0] if len(stop_ids) == 1 else None,
                "stop_ids": stop_ids,
                "route_row_ids": interaction.selected_route_row_ids,
            }
            save_decisions(decisions_path, decisions)
            if review_mismatches_path and decision_key in review_mismatches:
                review_mismatches.pop(decision_key, None)
                save_decisions(review_mismatches_path, review_mismatches)
            break

        if advance_to_next:
            stop_index += 1

    output_rows = build_output_rows(route_rows, route_assignments)
    osm_no_candidates = build_osm_no_candidates_rows(osm_stops, decisions)

    route_unmatched = []
    for route_row_id, row in enumerate(output_rows):
        if row["geometry"]:
            continue
        route_unmatched.append(
            {
                "route_row_id": str(route_row_id),
                "stop_name": row[route_stop_col],
                "route": row.get("route", ""),
                "direction": row.get("direction", ""),
                "transport": row.get("transport", ""),
            }
        )

    return output_rows, osm_no_candidates, route_unmatched, decisions


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm", default="osm_stops.csv", help="Path to OSM stops CSV.")
    parser.add_argument("--routes", default="easyway_routes.csv", help="Path to kyiv-routes CSV.")
    parser.add_argument("--route-stop-col", default="stop_name", help="Column in routes CSV with stop names.")
    parser.add_argument("--output", default="easyway-routes-with-geometry.csv", help="Output CSV path.")
    parser.add_argument(
        "--osm-unmatched-output",
        default="osm-stops-no-candidates.csv",
        help="OSM stops for which no candidate names were found in kyiv-routes.",
    )
    parser.add_argument(
        "--route-unmatched-output",
        default="easyway-routes-without-geometry.csv",
        help="Route rows still left without geometry after the run.",
    )
    parser.add_argument(
        "--decisions",
        default="stop-match-decisions.json",
        help="JSON cache for reviewed OSM rows and assigned route row ids.",
    )
    parser.add_argument(
        "--merge-decisions-from",
        default="",
        help="Merge another stop-match-decisions.json into --decisions and exit.",
    )
    parser.add_argument(
        "--review-mismatches",
        nargs="?",
        const="stop-match-review-mismatches.json",
        default="",
        help="Path to stop-match-review-mismatches.json; if set, revisit only these OSM rows.",
    )
    parser.add_argument(
        "--revisit-skipped",
        action="store_true",
        help="Reopen decisions with status='skipped' instead of skipping them.",
    )
    parser.add_argument("--candidate-limit", type=int, default=7, help="How many similar route stop names to show.")
    parser.add_argument(
        "--start-osm-row-id",
        type=int,
        default=0,
        help="Start reviewing from this OSM row id.",
    )
    parser.add_argument(
        "--auto-select-exact",
        action="store_true",
        help="Automatically skip name selection when there is exactly one exact candidate match.",
    )
    parser.add_argument(
        "--preview-map",
        default="current_stop_preview.html",
        help="HTML file overwritten with a map preview for the current OSM stop. Use '' to disable.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the current stop map in a browser for each new OSM stop.",
    )
    parser.add_argument(
        "--preview-browser-app",
        default="",
        help='Browser mode for preview URLs. Leave empty to use a pyppeteer-managed window, or set "system" to use the OS default browser.',
    )
    parser.add_argument(
        "--eway-browser-app",
        default="",
        help='Browser mode for eway URLs. Leave empty to use a pyppeteer-managed window, or set "system" to use the OS default browser.',
    )
    parser.add_argument(
        "--pyppeteer-executable-path",
        default="",
        help="Optional executable path for the browser that pyppeteer should launch. Leave empty to use bundled Chromium.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    PYPPETEER_BROWSER_MANAGER.executable_path = args.pyppeteer_executable_path or None

    if args.merge_decisions_from:
        merged_decisions, added_count, kept_count, conflicts = merge_decisions(
            base_path=Path(args.decisions),
            incoming_path=Path(args.merge_decisions_from),
        )
        save_decisions(Path(args.decisions), merged_decisions)
        print(f"Додано нових записів: {added_count}")
        print(f"Пропущено через однаковий status: {kept_count}")
        if conflicts:
            print("Конфлікти status:")
            for conflict in conflicts:
                print(f"  {conflict}")
        else:
            print("Конфліктів status немає.")
        return 0

    osm_rows = read_csv_rows(Path(args.osm))
    route_rows = read_csv_rows(Path(args.routes))
    if not route_rows:
        raise SystemExit("routes CSV is empty")
    if args.route_stop_col not in route_rows[0]:
        raise SystemExit(f'Column "{args.route_stop_col}" not found in {args.routes}')

    review_osm_row_ids: Optional[Set[str]] = None
    review_mismatches_path: Optional[Path] = None
    if args.review_mismatches:
        review_mismatches_path = Path(args.review_mismatches)
        review_decisions = load_decisions(review_mismatches_path)
        review_osm_row_ids = {str(key) for key in review_decisions.keys()}
        print(f"Повторний перегляд тільки для OSM row id: {len(review_osm_row_ids)}")

    preview_map_path = Path(args.preview_map) if args.preview_map else None
    output_rows, osm_no_candidates, route_unmatched, decisions = run_matching(
        osm_rows=osm_rows,
        route_rows=route_rows,
        route_stop_col=args.route_stop_col,
        decisions_path=Path(args.decisions),
        candidate_limit=args.candidate_limit,
        preview_map_path=preview_map_path,
        auto_select_exact=args.auto_select_exact,
        open_browser=args.open_browser,
        preview_browser_app=args.preview_browser_app or None,
        eway_browser_app=args.eway_browser_app or None,
        start_osm_row_id=args.start_osm_row_id,
        review_osm_row_ids=review_osm_row_ids,
        review_mismatches_path=review_mismatches_path,
        revisit_skipped=args.revisit_skipped,
    )

    write_csv(Path(args.output), output_rows, list(output_rows[0].keys()))
    write_csv(Path(args.osm_unmatched_output), osm_no_candidates, ["osm_row_id", "name", "geometry"])
    write_csv(
        Path(args.route_unmatched_output),
        route_unmatched,
        ["route_row_id", "stop_name", "route", "direction", "transport"],
    )

    matched_rows = sum(1 for row in output_rows if row["geometry"])
    print(f"Збережено: {args.output}")
    print(f"OSM без кандидатів: {args.osm_unmatched_output}")
    print(f"Route rows без geometry: {args.route_unmatched_output}")
    print(f"Опрацьовано OSM rows: {len(decisions)} / {len(osm_rows)}")
    print(f"Рядків kyiv-routes з geometry: {matched_rows} / {len(output_rows)}")
    print(f"OSM без кандидатів: {len(osm_no_candidates)}")
    print(f"Route rows без geometry: {len(route_unmatched)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
