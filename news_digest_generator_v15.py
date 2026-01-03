#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py (robust static generator)
- Single static index.html (no client-side fetching)
- Uses Google News RSS search + optional LLM structuring
- Embeds digest JSON into HTML template via __DIGEST_JSON__ placeholder
- Designed for GitHub Actions (handles SIGUSR1 stack dump safely)

Config schema (digest_config_v15.json):
{
  "title": "Ben的每日资讯简报",
  "views": [{"id":"global","label":"全球（不含中国）"},{"id":"cn","label":"中国相关"}],
  "kpis": [{"id":"DGS10","label":"US 10Y","unit":"%","source":"FRED"}, ...],
  "sections": [
     {"id":"macro","name":"宏观 ...","queries":[...],"hl":"en-US","gl":"US","ceid":"US:en"},
     ...
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import random
import re
import signal
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- make SIGUSR1 print stacks (and NOT kill the process) ---
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
    # SIGUSR1 is used by the workflow watchdog for stack dumps
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except Exception:
    pass

try:
    import requests  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: requests. Install with `pip install requests`.") from e

try:
    from jinja2 import Environment, BaseLoader
except Exception as e:
    raise RuntimeError("Missing dependency: jinja2. Install with `pip install jinja2`.") from e


BJT = dt.timezone(dt.timedelta(hours=8))

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)

DEFAULT_TIMEOUT = (10, 45)  # connect, read


def now_bjt() -> dt.datetime:
    return dt.datetime.now(tz=BJT)


def ymd(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def safe_json_extract(s: str) -> str:
    """Best-effort extraction of a JSON object from a messy LLM output."""
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    i = s.find("{")
    j = s.rfind("}")
    if 0 <= i < j:
        return s[i : j + 1]
    return s  # let json.loads fail


def slugify(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:80] or "item"


def truncate(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def http_get(url: str, *, timeout: Tuple[int, int] = DEFAULT_TIMEOUT, headers: Optional[Dict[str, str]] = None) -> str:
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    # requests will guess encoding; keep it simple
    r.encoding = r.encoding or "utf-8"
    return r.text


def http_get_bytes(url: str, *, timeout: Tuple[int, int] = DEFAULT_TIMEOUT, headers: Optional[Dict[str, str]] = None) -> bytes:
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    return r.content


def retry(
    fn,
    *,
    attempts: int = 3,
    base_sleep: float = 1.0,
    jitter: float = 0.3,
    name: str = "op",
    no_retry_status: Tuple[int, ...] = (400, 401, 403, 404),
):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except requests.HTTPError as e:
            last = e
            status = getattr(e.response, "status_code", None)
            # do not retry certain 4xx errors
            if status in no_retry_status and status != 429:
                raise
        except Exception as e:
            last = e
        sleep = base_sleep * (1.6 ** i) + random.random() * jitter
        print(f"[digest] retry {name}: {type(last).__name__}: {last} (sleep {sleep:.1f}s)", flush=True)
        time.sleep(sleep)
    raise last  # type: ignore


def parse_rss(xml_text: str) -> List[Dict[str, Any]]:
    """Parse RSS 2.0 feeds (Google News RSS included) using stdlib only."""
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    # RSS: <rss><channel><item>...
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        source_el = it.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        items.append({"title": title, "link": link, "pubDate": pub, "source": source})
    return items


def google_news_rss_url(query: str, hl: str, gl: str, ceid: str) -> str:
    q = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def fetch_google_news_items(query: str, *, hl: str, gl: str, ceid: str, limit: int) -> List[Dict[str, Any]]:
    url = google_news_rss_url(query, hl, gl, ceid)
    xml_text = retry(lambda: http_get(url), name="gnews", attempts=3)
    items = parse_rss(xml_text)
    # de-dupe by link
    seen = set()
    out = []
    for it in items:
        link = it.get("link") or ""
        if not link or link in seen:
            continue
        seen.add(link)
        out.append(it)
        if len(out) >= limit:
            break
    return out


def fred_csv_url(series_id: str) -> str:
    # No API key required for this endpoint (public csv)
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(series_id)}"


def parse_fred_csv(csv_bytes: bytes, max_points: int = 60) -> List[Tuple[str, Optional[float]]]:
    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.reader(text.splitlines())
    rows = list(reader)
    if len(rows) <= 1:
        return []
    # header: DATE,VALUE
    pts: List[Tuple[str, Optional[float]]] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        date_s = row[0].strip()
        val_s = row[1].strip()
        try:
            val = float(val_s) if val_s not in (".", "") else None
        except ValueError:
            val = None
        pts.append((date_s, val))
    # keep last max_points with non-null preference
    return pts[-max_points:]


def sparkline_svg(values: List[Optional[float]], width: int = 120, height: int = 28) -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        # empty sparkline placeholder
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"></svg>'
    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    # map to points
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        if v is None:
            continue
        x = (i / max(1, n - 1)) * (width - 2) + 1
        y = (1 - (v - vmin) / (vmax - vmin)) * (height - 2) + 1
        pts.append((x, y))
    if len(pts) < 2:
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"></svg>'
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{d}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />'
        f"</svg>"
    )


def format_number(x: Optional[float], decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(x)


@dataclass
class KPI:
    id: str
    label: str
    unit: str
    source: str
    value: Optional[float]
    delta: Optional[float]
    pct: Optional[float]
    series: List[Tuple[str, Optional[float]]]
    spark_svg: str

    @property
    def value_str(self) -> str:
        return format_number(self.value, 2)

    @property
    def delta_str(self) -> str:
        if self.delta is None:
            return "NA"
        return format_number(self.delta, 2)

    @property
    def pct_str(self) -> str:
        if self.pct is None:
            return "NA"
        return format_number(self.pct, 2)


def build_kpis(kpis_cfg: List[Dict[str, Any]], max_points: int = 60) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for k in kpis_cfg:
        # Support both "id"/"series" and "label"/"title" for backwards compatibility
        sid = k.get("id") or k.get("series", "")
        label = k.get("label") or k.get("title", sid)
        unit = k.get("unit", "")
        source = k.get("source", "FRED")
        if not sid:
            continue

        def _fetch():
            url = fred_csv_url(sid)
            b = http_get_bytes(url, timeout=(10, 35))
            return b

        try:
            b = retry(_fetch, name=f"fred {sid}", attempts=3, base_sleep=1.0, jitter=0.4, no_retry_status=(400, 401, 403, 404))
            series = parse_fred_csv(b, max_points=max_points)
        except Exception as e:
            print(f"[digest] kpi failed {sid}: {type(e).__name__}: {e}", flush=True)
            series = []

        vals = [v for _, v in series]
        # last non-null
        last = next((v for v in reversed(vals) if v is not None), None)
        prev = next((v for v in reversed(vals[:-1]) if v is not None), None) if len(vals) >= 2 else None
        delta = (last - prev) if (last is not None and prev is not None) else None
        pct = ((delta / prev) * 100.0) if (delta is not None and prev not in (None, 0.0)) else None

        # sparkline uses numeric series with None placeholders
        svg = sparkline_svg(vals[-30:], width=120, height=28)
        # determine direction for CSS class
        direction = "up" if (delta is not None and delta > 0) else ("down" if (delta is not None and delta < 0) else "flat")

        out.append(
            {
                "id": sid,
                "label": label,
                "unit": unit,
                "source": source,
                "value": last,
                "value_str": format_number(last, 2),
                "delta": delta,
                "delta_str": format_number(delta, 2) if delta is not None else "NA",
                "pct": pct,
                "pct_str": format_number(pct, 2) if pct is not None else "NA",
                "spark_svg": svg,
                "direction": direction,
                "series": series[-30:],
            }
        )
    return out


def openai_chat_json(model: str, system: str, user: str, *, timeout: Tuple[int, int] = (10, 60), max_tokens: int = 900) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": USER_AGENT}
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # JSON enforcement (supported by many OpenAI chat models)
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    def _do():
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(safe_json_extract(content))

    return retry(_do, name="openai", attempts=3, base_sleep=1.0, jitter=0.4, no_retry_status=(400, 401, 403, 404))


def guess_region(text_: str) -> str:
    t = (text_ or "").lower()
    cn_kw = ["china", "chinese", "beijing", "shanghai", "shenzhen", "hong kong", "taiwan", "prc", "cn", "cny", "pbo", "pboc"]
    if any(k in t for k in cn_kw) or re.search(r"[\u4e00-\u9fff]", text_ or ""):
        return "cn"
    return "global"


def build_raw_items_for_section(sc: Dict[str, Any], limit_raw: int) -> List[Dict[str, Any]]:
    hl = sc.get("hl", "en-US")
    gl = sc.get("gl", "US")
    ceid = sc.get("ceid", "US:en")
    queries = sc.get("queries", []) or []
    raw: List[Dict[str, Any]] = []
    for q in queries:
        q = str(q).strip()
        if not q:
            continue
        try:
            items = fetch_google_news_items(q, hl=hl, gl=gl, ceid=ceid, limit=limit_raw)
            print(f"[digest] gnews {sc.get('id')} q='{truncate(q, 24)}' entries={len(items)}", flush=True)
            raw.extend(items)
        except Exception as e:
            print(f"[digest] gnews {sc.get('id')} failed: {type(e).__name__}: {e}", flush=True)

    # dedupe across queries
    seen = set()
    out = []
    for it in raw:
        link = it.get("link") or ""
        if not link or link in seen:
            continue
        seen.add(link)
        out.append(it)
    return out[: max(limit_raw, 1)]


def llm_section_pack(model: str, section: Dict[str, Any], raw_items: List[Dict[str, Any]], items_per_section: int) -> Dict[str, Any]:
    """One LLM call per section to produce:
    - brief_zh: 300-500 chars (Chinese)
    - events: list of 10-20 event cards (conclusion-first)
    """
    sid = section.get("id", "")
    sname = section.get("name", sid)
    # keep prompt small
    candidates = []
    for it in raw_items[: min(len(raw_items), 40)]:
        candidates.append(
            {
                "title": truncate(it.get("title", ""), 160),
                "source": truncate(it.get("source", ""), 60),
                "pubDate": truncate(it.get("pubDate", ""), 60),
                "link": it.get("link", ""),
            }
        )
    system = (
        "You are a financial/geopolitical/tech digest editor. "
        "Return ONLY valid JSON object. No markdown, no extra text. "
        "Write in Simplified Chinese."
    )
    user = {
        "task": "Pack candidates into conclusion-first event cards and a section brief.",
        "constraints": {
            "section_brief_chars": [300, 500],
            "events_count": [max(10, min(20, items_per_section)), max(10, min(20, items_per_section))],
            "event_fields_required": ["id", "title_zh", "one_liner_zh", "facts", "why_it_matters", "watchlist", "source_hint", "date", "region", "links"],
            "facts_count": [3, 5],
            "links_count": [1, 3],
            "region_values": ["global", "cn"],
        },
        "section": {"id": sid, "name": sname},
        "candidates": candidates,
    }
    try:
        data = openai_chat_json(model, system, json.dumps(user, ensure_ascii=False), max_tokens=1200)
        # minimal normalization / defaults
        brief_zh = str(data.get("brief_zh", "")).strip()
        events = data.get("events", [])
        if not isinstance(events, list):
            events = []
        norm_events = []
        for i, ev in enumerate(events[:items_per_section]):
            if not isinstance(ev, dict):
                continue
            links = ev.get("links") or []
            if not isinstance(links, list):
                links = []
            links2 = []
            for lk in links[:3]:
                if isinstance(lk, dict) and lk.get("url"):
                    links2.append({"title": truncate(str(lk.get("title", "")), 60), "url": str(lk["url"])})
            if not links2:
                # fallback: attach candidate by index
                if i < len(candidates):
                    links2 = [{"title": candidates[i]["title"], "url": candidates[i]["link"]}]
            facts = ev.get("facts") or []
            if not isinstance(facts, list):
                facts = []
            facts2 = [truncate(str(x), 60) for x in facts[:5] if str(x).strip()]
            norm_events.append(
                {
                    "id": ev.get("id") or f"{sid}-{i+1}",
                    "title_zh": truncate(str(ev.get("title_zh", "")) or truncate(candidates[i]["title"] if i < len(candidates) else sname, 40), 60),
                    "one_liner_zh": truncate(str(ev.get("one_liner_zh", "")), 120),
                    "facts": facts2,
                    "why_it_matters": truncate(str(ev.get("why_it_matters", "")), 180),
                    "watchlist": truncate(str(ev.get("watchlist", "")), 160),
                    "source_hint": truncate(str(ev.get("source_hint", "")) or (candidates[i]["source"] if i < len(candidates) else ""), 60),
                    "date": truncate(str(ev.get("date", "")) or (candidates[i]["pubDate"] if i < len(candidates) else ""), 40),
                    "region": ev.get("region") if ev.get("region") in ("cn", "global") else guess_region(ev.get("title_zh", "") + " " + ev.get("one_liner_zh", "")),
                    "links": links2,
                }
            )
        if len(brief_zh) < 120:
            # LLM sometimes returns empty/too short brief
            brief_zh = fallback_section_brief(sname, raw_items)
        if not norm_events:
            norm_events = fallback_events(sid, sname, raw_items, items_per_section)
        tags = data.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [truncate(str(t), 16) for t in tags[:8] if str(t).strip()]
        return {"id": sid, "name": sname, "brief_zh": brief_zh, "tags": tags, "events": norm_events}
    except Exception as e:
        print(f"[digest] llm section pack failed {sid}: {type(e).__name__}: {e}", flush=True)
        return {"id": sid, "name": sname, "brief_zh": fallback_section_brief(sname, raw_items), "tags": [], "events": fallback_events(sid, sname, raw_items, items_per_section)}


def fallback_section_brief(section_name: str, raw_items: List[Dict[str, Any]]) -> str:
    titles = [truncate(it.get("title", ""), 70) for it in raw_items[:8]]
    # 300-500 chars: heuristic paragraph + bullets in-line
    base = f"{section_name}方面，今日信息密度主要集中在以下几条主线："
    bullets = "；".join([t for t in titles if t]) or "暂无足够候选条目，建议提高抓取上限或更换关键词。"
    tail = "。建议把注意力放在“政策/监管口径是否变化”“资金/库存/产能数据是否验证叙事”“事件是否会在48小时内升级”三点。"
    txt = base + bullets + tail
    return truncate(txt, 480)


def fallback_events(section_id: str, section_name: str, raw_items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out = []
    for i, it in enumerate(raw_items[:n]):
        title = it.get("title", "") or section_name
        out.append(
            {
                "id": f"{section_id}-{i+1}",
                "title_zh": truncate(title, 60),
                "one_liner_zh": truncate(title, 120),
                "facts": [],
                "why_it_matters": "（自动降级）模型输出不可用，本条目为候选标题占位。请点击原文核对。",
                "watchlist": "关注后续官方更新/数据披露与二级市场反应。",
                "source_hint": truncate(it.get("source", ""), 60),
                "date": truncate(it.get("pubDate", ""), 40),
                "region": guess_region(title),
                "links": [{"title": truncate(title, 60), "url": it.get("link", "")}],
            }
        )
    # if insufficient, pad
    while len(out) < n:
        out.append(
            {
                "id": f"{section_id}-{len(out)+1}",
                "title_zh": f"{section_name}（占位）",
                "one_liner_zh": "暂无足够候选条目。",
                "facts": [],
                "why_it_matters": "请调整关键词或提高抓取上限。",
                "watchlist": "—",
                "source_hint": "",
                "date": "",
                "region": "global",
                "links": [],
            }
        )
    return out


def llm_top_brief(model: str, title: str, sections: List[Dict[str, Any]], kpis: List[Dict[str, Any]]) -> str:
    # compact inputs
    sec_lines = []
    for s in sections:
        evs = s.get("events", []) or []
        head = truncate(s.get("name", s.get("id", "")), 18)
        top_titles = [truncate(ev.get("title_zh", ""), 18) for ev in evs[:3] if isinstance(ev, dict)]
        sec_lines.append(f"{head}: " + " / ".join([t for t in top_titles if t]))
    kpi_lines = []
    for k in kpis[:9]:
        kpi_lines.append(f"{k.get('label')} {k.get('value_str')}{k.get('unit')} Δ{k.get('delta_str')}")

    system = (
        "You are an editor writing the '今日要点' for a Chinese daily intelligence brief. "
        "Return ONLY a JSON object: {\"top_brief_zh\": \"...\"}. "
        "Keep it 300-500 Chinese characters. Concrete, non-empty, actionable."
    )
    user = {
        "title": title,
        "constraints": {"chars": [300, 500], "tone": "结论优先，给出变量与观察点，避免空泛。"},
        "inputs": {"section_heads": sec_lines, "kpis": kpi_lines},
        "structure_hint": [
            "1) 风险偏好/利率/美元的方向性判断 + 触发条件",
            "2) 地缘/制裁的合规要点（对交易/供应链的实际影响）",
            "3) 算力/数据中心供给链（GPU/电力/冷却）与资本开支信号",
            "4) 金属/能源/碳的关键价格-库存-政策组合",
            "5) 东南亚产能迁移/关键矿产与贸易摩擦的落点",
        ],
    }
    try:
        data = openai_chat_json(model, system, json.dumps(user, ensure_ascii=False), max_tokens=900)
        s = str(data.get("top_brief_zh", "")).strip()
        if 300 <= len(s) <= 520:
            return s
        # fallback to heuristic if too short
    except Exception as e:
        print(f"[digest] llm daily brief failed: {type(e).__name__}: {e}", flush=True)
    # heuristic fallback
    parts = []
    if kpi_lines:
        parts.append("市场层面：" + "；".join(kpi_lines[:4]) + "。")
    if sec_lines:
        parts.append("主题层面：" + "；".join(sec_lines[:4]) + "。")
    parts.append("建议聚焦：利率与美元方向、主要制裁/出口管制口径、数据中心供电与液冷扩张节奏、关键金属库存/供给扰动、东南亚产能迁移与关税摩擦。")
    return truncate("".join(parts), 500)


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # sections: allow list or dict
    secs = cfg.get("sections")
    if isinstance(secs, list):
        sec_dict = {}
        for s in secs:
            if isinstance(s, dict) and s.get("id"):
                sec_dict[s["id"]] = s
        cfg["sections"] = sec_dict
    elif secs is None:
        cfg["sections"] = {}
    # views default
    if not cfg.get("views"):
        cfg["views"] = [{"id": "global", "label": "全球（不含中国）"}, {"id": "cn", "label": "中国相关"}]
    if not cfg.get("title"):
        cfg["title"] = "Ben的每日资讯简报"
    return cfg


def build_digest(cfg: Dict[str, Any], *, date_str: str, model: str, limit_raw: int, items_per_section: int) -> Dict[str, Any]:
    cfg = normalize_config(cfg)
    title = cfg.get("title", "Ben的每日资讯简报")

    print(f"[digest] date={date_str} model={model} limit_raw={limit_raw} items={items_per_section}", flush=True)

    print(f"[digest] kpi start: {len(cfg.get('kpis', []) or [])}", flush=True)
    kpis = build_kpis(cfg.get("kpis", []) or [])
    print(f"[digest] kpi done: {len(kpis)}", flush=True)

    sections_cfg: Dict[str, Any] = cfg.get("sections", {}) or {}
    sections_out: List[Dict[str, Any]] = []
    print(f"[digest] sections start: {len(sections_cfg)}", flush=True)
    for sid, sc in sections_cfg.items():
        name = sc.get("name", sid)
        print(f"[digest] section start: {sid} ({name})", flush=True)
        raw_items = build_raw_items_for_section({"id": sid, **sc}, limit_raw=limit_raw)
        # If no raw items, still return placeholders
        packed = llm_section_pack(model, {"id": sid, "name": name}, raw_items, items_per_section)
        sections_out.append(packed)
        print(f"[digest] section done: {sid} events={len(packed.get('events', []) or [])}", flush=True)
    print(f"[digest] sections done: {len(sections_out)}", flush=True)

    top_brief = llm_top_brief(model, title, sections_out, kpis)

    digest = {
        "title": title,
        "date": date_str,
        "generated_at_bjt": now_bjt().strftime("%Y-%m-%d %H:%M (BJT)"),
        "views": cfg.get("views", []),
        "kpis": kpis,
        "sections": sections_out,
        "top_brief_zh": top_brief,
        "meta": {"limit_raw": limit_raw, "items_per_section": items_per_section, "model": model},
    }
    return digest


def render_html(template_path: str, digest: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Render the HTML template using Jinja2."""
    tpl_text = open(template_path, "r", encoding="utf-8").read()
    
    # Check if template uses Jinja2 syntax or __DIGEST_JSON__ placeholder
    uses_jinja2 = "{% for" in tpl_text or "{{ VIEWS" in tpl_text or "{{ views" in tpl_text
    
    if not uses_jinja2:
        # Fallback to old JSON embedding method
        date_str = str(digest.get("date", ""))
        tpl_text = tpl_text.replace("{{ DATE }}", html.escape(date_str))
        tpl_text = tpl_text.replace("{{DATE}}", html.escape(date_str))
        title = str(digest.get("title", "Ben的每日资讯简报"))
        tpl_text = re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)} · {html.escape(date_str)}</title>", tpl_text, flags=re.I | re.S)
        dj = json.dumps(digest, ensure_ascii=False)
        dj = dj.replace("</script>", "<\\/script>")
        if "__DIGEST_JSON__" in tpl_text:
            tpl_text = tpl_text.replace("__DIGEST_JSON__", dj)
        return tpl_text
    
    # Use Jinja2 for rendering
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(tpl_text)
    
    # Build template context
    date_str = digest.get("date", "")
    title = digest.get("title", "Ben的每日资讯简报")
    generated_at = digest.get("generated_at_bjt", "")
    
    # Transform KPIs to template format
    kpis_out = []
    for k in digest.get("kpis", []):
        value = k.get("value")
        delta = k.get("delta")
        pct = k.get("pct")
        
        # Determine direction
        direction = "flat"
        if delta is not None:
            if delta > 0.0001:
                direction = "up"
            elif delta < -0.0001:
                direction = "down"
        
        kpis_out.append({
            "id": k.get("id", ""),
            "title": k.get("label", k.get("id", "")),
            "unit": k.get("unit", ""),
            "value": value,
            "delta_abs": delta,
            "delta_pct": pct,
            "direction": direction,
            "spark": k.get("spark_svg", ""),
            "source": k.get("source", "FRED"),
        })
    
    # Build alerts from config
    alerts_out = []
    for a in cfg.get("alerts", []) or []:
        sid = a.get("series", "")
        # Find matching KPI value
        kpi_val = None
        kpi_title = sid
        for k in kpis_out:
            if k["id"] == sid:
                kpi_val = k["value"]
                kpi_title = k["title"]
                break
        
        if kpi_val is None:
            continue
        
        threshold = a.get("value", 0)
        op = a.get("op", ">")
        triggered = False
        
        if op == ">" and kpi_val > threshold:
            triggered = True
        elif op == "<" and kpi_val < threshold:
            triggered = True
        elif op == ">=" and kpi_val >= threshold:
            triggered = True
        elif op == "<=" and kpi_val <= threshold:
            triggered = True
        elif op == "abs>" and abs(kpi_val) > threshold:
            triggered = True
        
        if triggered:
            alerts_out.append({
                "severity": a.get("severity", "info"),
                "title": kpi_title,
                "op": op,
                "threshold": threshold,
                "message": a.get("message", ""),
            })
    
    # Build views structure
    sections = digest.get("sections", [])
    top_brief = digest.get("top_brief_zh", "")
    
    # Helper to build top/brief structure
    def make_top(brief_text: str) -> Dict[str, Any]:
        return {
            "one_liner": brief_text[:100] if brief_text else "",
            "facts": [],
            "actions": [],
            "watch": [],
            "summary": brief_text,
        }
    
    def make_brief(section: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "one_liner": section.get("brief_zh", "")[:100] if section.get("brief_zh") else "",
            "facts": [],
            "impacts": [],
            "actions": [],
            "watch": [],
            "summary": section.get("brief_zh", ""),
        }
    
    def transform_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform events to template format."""
        out = []
        for ev in events:
            out.append({
                "id": ev.get("id", ""),
                "title": ev.get("title_zh", ev.get("title", "")),
                "headline_zh": ev.get("title_zh", ev.get("title", "")),
                "conclusion": ev.get("one_liner_zh", ev.get("summary_zh", "")),
                "tags": ev.get("tags", []),
                "region": ev.get("region", "global"),
                "cn_related": ev.get("region") == "cn",
                "evidence": ev.get("facts", []),
                "facts": ev.get("facts", []),
                "impact": [ev.get("why_it_matters", "")] if ev.get("why_it_matters") else [],
                "actions": [],
                "next": [ev.get("watchlist", "")] if ev.get("watchlist") else [],
                "watch": [ev.get("watchlist", "")] if ev.get("watchlist") else [],
                "sources": ev.get("links", []),
                "links": ev.get("links", []),
            })
        return out
    
    # Build sections with transformed data
    sections_all = []
    sections_cn = []
    sections_global = []
    
    for s in sections:
        events = transform_events(s.get("events", []))
        section_data = {
            "id": s.get("id", ""),
            "name": s.get("name", ""),
            "brief": make_brief(s),
            "events": events,
            "cards": events,  # Alias for template compatibility
        }
        sections_all.append(section_data)
        
        # Filter for CN view
        cn_events = [e for e in events if e.get("cn_related") or e.get("region") == "cn"]
        if cn_events:
            sections_cn.append({**section_data, "events": cn_events, "cards": cn_events})
        
        # Filter for global view
        global_events = [e for e in events if not e.get("cn_related") and e.get("region") != "cn"]
        if global_events:
            sections_global.append({**section_data, "events": global_events, "cards": global_events})
    
    # Build nav for each view
    def make_nav(secs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"id": s["id"], "name": s["name"], "count": len(s.get("events", []))} for s in secs]
    
    views = {
        "all": {
            "key": "all",
            "label": "全部",
            "top": make_top(top_brief),
            "sections": sections_all,
            "nav": make_nav(sections_all),
        },
        "cn": {
            "key": "cn",
            "label": "中国相关",
            "top": make_top(f"（中国相关）{top_brief}"),
            "sections": sections_cn,
            "nav": make_nav(sections_cn),
        },
        "global": {
            "key": "global",
            "label": "全球（不含中国）",
            "top": make_top(f"（全球）{top_brief}"),
            "sections": sections_global,
            "nav": make_nav(sections_global),
        },
    }
    
    # Template context
    context = {
        "TITLE": title,
        "DATE": date_str,
        "GENERATED_AT": generated_at,
        "KPIS": kpis_out,
        "ALERTS": alerts_out,
        "VIEWS": views,
        "views": views,  # lowercase for Jinja2 iteration
    }
    
    return template.render(**context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD, default today (BJT)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items_per_section", type=int, default=15)
    args = ap.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = ymd(now_bjt().date())

    cfg = json.loads(open(args.config, "r", encoding="utf-8").read())
    print("[digest] boot", flush=True)

    digest = build_digest(cfg, date_str=date_str, model=args.model, limit_raw=args.limit_raw, items_per_section=args.items_per_section)

    print("[digest] render html", flush=True)
    out_html = render_html(args.template, digest, cfg)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_html)
    print(f"[digest] wrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()
