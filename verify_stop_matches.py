#!/usr/bin/env python3
"""Review matched stops from stop-match-decisions.json in two browser windows."""

from __future__ import annotations

import argparse
import atexit
import asyncio
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty

try:
    from pyppeteer import launch
except ImportError:
    launch = None  # type: ignore[assignment]

from match_stops import load_decisions, save_decisions, stop_id_link, write_preview_map


class ReviewBrowserManager:
    def __init__(self, executable_path: Optional[str] = None) -> None:
        self.executable_path = executable_path or None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.preview_browser = None
        self.eway_browser = None
        self.preview_page = None
        self.eway_pages: List[object] = []
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

        thread = threading.Thread(target=runner, name="verify-stop-matches-browser-loop", daemon=True)
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

    async def _async_start(self) -> bool:
        if self.started:
            return True
        self.preview_browser = await self._launch_browser()
        self.eway_browser = await self._launch_browser()
        if self.preview_browser is None or self.eway_browser is None:
            return False
        self.preview_page = await self.preview_browser.newPage()
        self.eway_pages = [await self.eway_browser.newPage()]
        self.started = True
        return True

    def _ensure_started(self) -> bool:
        if self.started:
            return True
        if launch is None:
            print(
                "Помилка: pyppeteer не встановлений. Виконай `pip install -r requirements.txt`.",
                file=sys.stderr,
            )
            return False
        self._start_loop()
        try:
            return self._run_coro(self._async_start())
        except Exception as exc:
            print(f"Помилка запуску браузера через pyppeteer: {exc}", file=sys.stderr)
            return False

    async def _async_open_preview(self, url: str) -> bool:
        if not url or not await self._async_start():
            return False
        self.preview_page = await self._async_ensure_page(self.preview_browser, self.preview_page)
        if self.preview_page is None:
            return False
        try:
            await self._async_goto(self.preview_page, url)
            await self.preview_page.bringToFront()
            return True
        except Exception:
            self.preview_page = await self.preview_browser.newPage()
            await self._async_goto(self.preview_page, url)
            await self.preview_page.bringToFront()
            return True

    def open_preview(self, url: str) -> bool:
        if not self._ensure_started():
            return False
        return self._run_coro(self._async_open_preview(url))

    async def _async_prepare_eway_pages(self, count: int) -> List[object]:
        if self.eway_browser is None:
            return []

        alive_pages: List[object] = []
        for page in self.eway_pages:
            try:
                if not page.isClosed():
                    alive_pages.append(page)
            except Exception:
                continue
        self.eway_pages = alive_pages

        while len(self.eway_pages) < count:
            self.eway_pages.append(await self.eway_browser.newPage())

        extra_pages = self.eway_pages[count:]
        self.eway_pages = self.eway_pages[:count]
        for page in extra_pages:
            try:
                await page.close()
            except Exception:
                pass

        if not self.eway_pages and count > 0:
            self.eway_pages.append(await self.eway_browser.newPage())
        return self.eway_pages

    async def _async_open_eway_links(self, urls: Sequence[str]) -> bool:
        clean_urls = [url for url in urls if url]
        if not clean_urls or not await self._async_start():
            return False

        pages = await self._async_prepare_eway_pages(len(clean_urls))
        if len(pages) != len(clean_urls):
            return False

        for page, url in zip(pages, clean_urls):
            try:
                await self._async_goto(page, url)
            except Exception:
                replacement_page = await self.eway_browser.newPage()
                index = pages.index(page)
                pages[index] = replacement_page
                self.eway_pages[index] = replacement_page
                await self._async_goto(replacement_page, url)

        try:
            await pages[0].bringToFront()
        except Exception:
            pass
        return True

    def open_eway_links(self, urls: Sequence[str]) -> bool:
        if not self._ensure_started():
            return False
        return self._run_coro(self._async_open_eway_links(urls))

    async def _async_close_eway_pages(self) -> bool:
        if not await self._async_start() or self.eway_browser is None:
            return False
        for page in list(self.eway_pages):
            try:
                if not page.isClosed():
                    await page.close()
            except Exception:
                pass
        self.eway_pages = []
        return True

    def close_eway_pages(self) -> bool:
        if not self._ensure_started():
            return False
        return self._run_coro(self._async_close_eway_pages())

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
        self.eway_pages = []
        self.started = False
        return True

    def close(self) -> None:
        if self.loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._async_close(), self.loop).result(timeout=10)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop = None
            self.thread = None


BROWSER_MANAGER = ReviewBrowserManager()
atexit.register(BROWSER_MANAGER.close)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        default="stop-match-decisions.json",
        help="Path to stop-match-decisions.json.",
    )
    parser.add_argument(
        "--review-output",
        default="stop-match-review-mismatches.json",
        help="Path to JSON with only mismatched review results.",
    )
    parser.add_argument(
        "--preview-map",
        default="current_stop_preview.html",
        help="HTML file overwritten with a preview for the current stop.",
    )
    parser.add_argument(
        "--start-osm-row-id",
        type=int,
        default=0,
        help="Start review from this OSM row id.",
    )
    parser.add_argument(
        "--pyppeteer-executable-path",
        default="",
        help="Optional executable path for the browser that pyppeteer should launch.",
    )
    parser.add_argument(
        "--revisit-skipped",
        action="store_true",
        help="Include decisions with status='skipped' in addition to matched ones.",
    )
    return parser.parse_args(argv)


def build_review_items(
    decisions: Dict[str, Dict[str, object]],
    start_osm_row_id: int,
    revisit_skipped: bool,
) -> List[Tuple[str, Dict[str, object]]]:
    items: List[Tuple[str, Dict[str, object]]] = []
    for decision_key, decision in decisions.items():
        status = str(decision.get("status") or "")
        if status != "matched" and not (revisit_skipped and status == "skipped"):
            continue
        if int(decision_key) < start_osm_row_id:
            continue
        items.append((decision_key, decision))
    items.sort(key=lambda item: int(item[0]))
    return items


def extract_stop_ids(decision: Dict[str, object]) -> List[str]:
    raw_stop_ids = decision.get("stop_ids")
    if isinstance(raw_stop_ids, list):
        result = []
        for value in raw_stop_ids:
            normalized = str(value).strip()
            if normalized and normalized not in result:
                result.append(normalized)
        if result:
            return result

    raw_stop_id = str(decision.get("stop_id") or "").strip()
    return [raw_stop_id] if raw_stop_id else []


def open_review_windows(preview_map_path: Path, decision_key: str, decision: Dict[str, object]) -> None:
    osm_name = str(decision.get("osm_name", ""))
    geometry = str(decision.get("geometry", ""))
    preview_path = write_preview_map(preview_map_path, osm_name, geometry)
    if preview_path is not None:
        if not BROWSER_MANAGER.open_preview(preview_path.resolve().as_uri()):
            print("Не вдалося відкрити preview-вікно.", file=sys.stderr)

    stop_ids = extract_stop_ids(decision)
    eway_links = [stop_id_link(stop_id) for stop_id in stop_ids if stop_id]
    eway_links = [link for link in eway_links if link]
    if eway_links:
        if not BROWSER_MANAGER.open_eway_links(eway_links):
            print("Не вдалося відкрити easyway-вікно.", file=sys.stderr)
    else:
        BROWSER_MANAGER.close_eway_pages()

    print()
    print(f"OSM row id: {decision_key}")
    print(f'OSM назва: "{osm_name}"')
    print(f"geometry: {geometry}")
    print(f'Назва пошуку: "{decision.get("search_name", "")}"')
    print(f'Вибрана назва: "{decision.get("candidate_name", "")}"')
    print(f"stop_ids: {', '.join(stop_ids) if stop_ids else 'немає'}")
    print(f"route_row_ids: {decision.get('route_row_ids', [])}")


def prompt_review_choice(index: int, total: int) -> str:
    prompt = f"[{index}/{total}] Співпадає? y/space = так, n = ні, b = назад, q = вийти: "
    while True:
        print(prompt, end="", flush=True)
        if os.name == "nt":
            raw = msvcrt.getwch().lower()
            print("space" if raw == " " else raw)
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                raw = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print("space" if raw == " " else raw)
        if raw == " ":
            return "y"
        if raw in {"y", "n", "b", "q"}:
            return raw
        print("Некоректний ввід.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    BROWSER_MANAGER.executable_path = args.pyppeteer_executable_path or None

    if launch is None:
        raise SystemExit("pyppeteer не встановлений. Виконай: pip install -r requirements.txt")

    decisions_path = Path(args.decisions)
    review_output_path = Path(args.review_output)
    preview_map_path = Path(args.preview_map)

    decisions = load_decisions(decisions_path)
    review_mismatches = load_decisions(review_output_path)
    review_items = build_review_items(decisions, args.start_osm_row_id, args.revisit_skipped)
    if not review_items:
        if args.revisit_skipped:
            print("Немає matched/skipped-записів для перевірки.")
        else:
            print("Немає matched-записів для перевірки.")
        return 0

    total = len(review_items)
    index = 0
    while index < total:
        decision_key, decision = review_items[index]
        open_review_windows(preview_map_path, decision_key, decision)
        answer = prompt_review_choice(index + 1, total)
        if answer == "q":
            break
        if answer == "b":
            if index > 0:
                index -= 1
            continue
        if answer == "y":
            if decision_key in review_mismatches:
                review_mismatches.pop(decision_key, None)
                save_decisions(review_output_path, review_mismatches)
        else:
            review_mismatches[decision_key] = {
                "status": "mismatch",
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                "osm_name": decision.get("osm_name"),
                "geometry": decision.get("geometry"),
                "search_name": decision.get("search_name"),
                "candidate_name": decision.get("candidate_name"),
                "stop_id": decision.get("stop_id"),
                "stop_ids": decision.get("stop_ids", []),
                "route_row_ids": decision.get("route_row_ids", []),
            }
            save_decisions(review_output_path, review_mismatches)
        index += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
