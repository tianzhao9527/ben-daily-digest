#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py (stabilized)
- Primary source: Google News RSS (no fragile official RSS endpoints by default)
- Robust OpenAI JSON handling + strict timeouts + deterministic fallback so the page never renders empty
- Template expects: DATE, DIGEST_JSON (and optionally TITLE)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import email.utils
import faulthandler
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from jinja2 import Template

# -------------------------
# Utilities / logging
# -------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def now_utc() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)

def parse_rfc2822(dt_str: str) -> _dt.datetime | None:
    try:
        dt = email.utils.parsedate_to_datetime(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc)
    except Exception:
        return None

def safe_json_dumps(obj) -> str:
    # Avoid </script> injection breaking the template.
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s.replace("</", "<\\/")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def retry(fn, *, tries=3, base_sleep=1.0, jitter=0.25, name="op"):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i == tries - 1:
                break
            sleep = base_sleep * (1.6 ** i) + random.random() * jitter
            log(f"[digest] retry {name}: {type(e).__name__}: {e} (sleep {sleep:.1f}s)")
            time.sleep(sleep)
    raise last

# -------------------------
# Config model
# -------------------------

@dataclasses.dataclass
class KPI:
    id: str
    name: str
    unit: str = ""
    value: float | None = None
    delta: float | None = None
    source: str | None = None
    updated_at: str | None = None

@dataclasses.dataclass
class RawItem:
    title: str
    url: str
    source: str
    published_at: str | None = None  # ISO date
    query: str | None = None

@dataclasses.dataclass
class Event:
    title_zh: str
    summary_zh: str
    region: str = "global"  # used by template: all/cn/us/eu/asia/global
    score: float = 0.5
    sources: list[dict] = dataclasses.field(default_factory=list)  # {title,url,source,published_at}

@dataclasses.dataclass
class Section:
    id: str
    name: str
    brief_zh: str = ""
    brief_en: str = ""
    tags: list[str] = dataclasses.field(default_factory=list)
    events: list[Event] = dataclasses.field(default_factory=list)

# -------------------------
# Google News RSS
# -------------------------

def google_news_rss_url(query: str, hl="en-US", gl="US", ceid="US:en") -> str:
    q = urllib.parse.quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

def fetch_rss(url: str, timeout=(6, 20)) -> str:
    resp = requests.get(url, headers={"User-Agent": "ben-digest-bot/1.0"}, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def parse_google_news_rss(xml_text: str, *, query: str | None, limit: int) -> list[RawItem]:
    # Google News returns RSS 2.0
    items: list[RawItem] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items
    channel = root.find("channel")
    if channel is None:
        return items

    for it in channel.findall("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        src_el = it.find("source")
        source = (src_el.text.strip() if src_el is not None and src_el.text else "Google News")
        dt = parse_rfc2822(pub)
        published_at = dt.date().isoformat() if dt else None

        if not title or not link:
            continue
        items.append(RawItem(title=title, url=link, source=source, published_at=published_at, query=query))
        if len(items) >= limit:
            break
    return items

def fetch_google_news_items(query: str, *, limit: int) -> list[RawItem]:
    url = google_news_rss_url(query)
    xml_text = retry(lambda: fetch_rss(url), name="gnews")
    return parse_google_news_rss(xml_text, query=query, limit=limit)

def dedupe_items(items: list[RawItem]) -> list[RawItem]:
    seen = set()
    out = []
    for it in items:
        key = sha1((it.title.lower().strip() + "|" + it.url.strip()))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

# -------------------------
# KPIs (FRED csv)
# -------------------------

def fred_csv(series_id: str) -> str:
    # public CSV endpoint
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote_plus(series_id)}"

def fetch_fred_latest(series_id: str) -> tuple[float | None, str | None]:
    url = fred_csv(series_id)
    def _do():
        r = requests.get(url, timeout=(6, 20))
        r.raise_for_status()
        return r.text
    csv_text = retry(_do, name=f"fred {series_id}", tries=3, base_sleep=1.0)

    # CSV columns: DATE,VALUE
    lines = [x.strip() for x in csv_text.splitlines() if x.strip()]
    if len(lines) < 2:
        return None, None
    # walk from bottom to find last numeric
    for row in reversed(lines[1:]):
        parts = row.split(",")
        if len(parts) < 2:
            continue
        date_s, val_s = parts[0].strip(), parts[1].strip()
        if val_s in (".", ""):
            continue
        try:
            return float(val_s), date_s
        except Exception:
            continue
    return None, None

def fetch_fred_recent(series_id: str, max_points: int = 40) -> tuple[list[float], list[str]]:
    """Fetch recent numeric observations from FRED graph CSV endpoint (no API key).
    Returns (values, dates) in chronological order, limited to last `max_points` numeric points.
    """
    url = fred_csv(series_id)
    def _do():
        r = requests.get(url, timeout=(6, 20))
        r.raise_for_status()
        return r.text
    csv_text = retry(_do, name=f"fred {series_id} recent", tries=3, base_sleep=1.0)

    lines = [x.strip() for x in csv_text.splitlines() if x.strip()]
    if len(lines) < 2:
        return [], []
    vals: list[float] = []
    dates: list[str] = []
    # parse all numeric then slice last N
    for row in lines[1:]:
        parts = row.split(",")
        if len(parts) < 2:
            continue
        dt, val_s = parts[0].strip(), parts[1].strip()
        if val_s in (".", ""):
            continue
        try:
            v = float(val_s)
        except Exception:
            continue
        dates.append(dt)
        vals.append(v)
    if not vals:
        return [], []
    if max_points and len(vals) > max_points:
        dates = dates[-max_points:]
        vals = vals[-max_points:]
    return vals, dates


def _color_for_key(key: str) -> str:
    """Deterministic color from key."""
    palette = ["#0a84ff", "#34c759", "#ff9f0a", "#ff453a", "#bf5af2", "#64d2ff", "#ffd60a", "#30d158", "#ff375f"]
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return palette[h % len(palette)]


def _spark_svg(values: list[float], stroke: str) -> str:
    if not values or len(values) < 2:
        return ""
    w, h, pad = 100.0, 44.0, 3.0
    mn, mx = min(values), max(values)
    span = (mx - mn) if (mx - mn) != 0 else 1.0
    pts = []
    for i, v in enumerate(values):
        x = pad + (w - 2 * pad) * (i / (len(values) - 1))
        y = pad + (h - 2 * pad) * (1.0 - ((v - mn) / span))
        pts.append((x, y))
    d = f"M {pts[0][0]:.2f},{pts[0][1]:.2f} " + " ".join([f"L {x:.2f},{y:.2f}" for x, y in pts[1:]])
    return f'<svg class="spark" viewBox="0 0 100 44" preserveAspectRatio="none" aria-hidden="true"><path d="{d}" stroke="{stroke}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'


# -------------------------
# OpenAI (chat.completions via requests)
# -------------------------

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def _openai_headers() -> dict:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def extract_first_json_object(text: str) -> dict | None:
    # Try strict first
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Try to extract a top-level {...}
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i+1]
                    try:
                        obj = json.loads(cand)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None

def openai_chat_json(model: str, messages: list[dict], *, timeout=(10, 45), max_tokens: int = 900) -> dict:
    payload_base = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    def _post(payload: dict) -> requests.Response:
        r = requests.post(OPENAI_URL, headers=_openai_headers(), data=json.dumps(payload), timeout=timeout)
        if r.status_code >= 400:
            # surface server error body for debugging
            raise requests.HTTPError(f"{r.status_code} {r.text[:400]}", response=r)
        return r

    # Attempt 1: use response_format to enforce JSON when supported
    try_payload = dict(payload_base)
    try_payload["response_format"] = {"type": "json_object"}

    try:
        resp = retry(lambda: _post(try_payload), name="openai", tries=3, base_sleep=1.1)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        obj = extract_first_json_object(content)
        if obj is None:
            raise ValueError("OpenAI did not return JSON object")
        return obj
    except Exception as e1:
        # Fallback: no response_format (some models/accounts reject it)
        resp = retry(lambda: _post(payload_base), name="openai(no_rf)", tries=3, base_sleep=1.1)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        obj = extract_first_json_object(content)
        if obj is None:
            raise ValueError(f"OpenAI did not return JSON object (fallback). Raw head: {content[:200]}")
        return obj

# -------------------------
# LLM prompts
# -------------------------

def llm_section_pack(model: str, section: Section, raw_items: list[RawItem], *, items_per_section: int) -> tuple[str, list[Event]]:
    # Keep prompt small: only send short fields
    lines = []
    for it in raw_items[: min(len(raw_items), max(12, items_per_section * 2))]:
        lines.append({
            "title": it.title,
            "source": it.source,
            "date": it.published_at,
            "url": it.url,
        })

    sys_prompt = (
        "你是一个严格输出 JSON 的新闻摘要助手。"
        "根据输入的新闻条目，生成中文摘要与事件卡。"
        "注意：必须返回一个 JSON object（不要 markdown、不要解释）。"
    )
    user_prompt = {
        "section": {"id": section.id, "name": section.name},
        "items": lines,
        "requirements": {
            "brief_zh": "200-400 字中文，先结论后依据，避免空话。",
            "events": f"生成 {items_per_section} 个事件卡，每个事件卡包含：title_zh(<=22字)、summary_zh(80-140字)、region(枚举: cn/us/eu/asia/global)、score(0-1)、sources(2-4条引用：title,url,source,date)。"
        }
    }

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]
    obj = openai_chat_json(model, messages, max_tokens=1200)

    brief_zh = (obj.get("brief_zh") or obj.get("brief") or "").strip()
    evs = obj.get("events") or []
    events: list[Event] = []
    if isinstance(evs, list):
        for e in evs[:items_per_section]:
            try:
                title_zh = str(e.get("title_zh") or e.get("title") or "").strip()
                summary_zh = str(e.get("summary_zh") or e.get("summary") or "").strip()
                region = str(e.get("region") or "global").strip()
                score = float(e.get("score") if e.get("score") is not None else 0.5)
                sources = e.get("sources") if isinstance(e.get("sources"), list) else []
                sources2 = []
                for s in sources[:4]:
                    if not isinstance(s, dict):
                        continue
                    sources2.append({
                        "title": str(s.get("title") or "")[:140],
                        "url": str(s.get("url") or ""),
                        "source": str(s.get("source") or ""),
                        "published_at": str(s.get("date") or s.get("published_at") or ""),
                    })
                if title_zh and summary_zh:
                    events.append(Event(title_zh=title_zh, summary_zh=summary_zh, region=region, score=score, sources=sources2))
            except Exception:
                continue

    # Hard fallback if LLM returns too few
    if len(events) < max(3, items_per_section // 3):
        raise ValueError(f"LLM returned too few events: {len(events)}")
    return brief_zh, events

# -------------------------
# Deterministic fallback (never empty)
# -------------------------

def fallback_pack(section: Section, raw_items: list[RawItem], *, items_per_section: int) -> tuple[str, list[Event]]:
    # Use top headlines as "events"; keep in Chinese light (minimal) to avoid blank page
    picked = raw_items[:items_per_section]
    events = []
    for it in picked:
        title = it.title.strip()
        summary = f"{it.source} 报道：{title}"
        events.append(Event(
            title_zh=title[:28],  # may be English; acceptable as fallback
            summary_zh=summary[:160],
            region="global",
            score=0.4,
            sources=[{"title": it.title[:140], "url": it.url, "source": it.source, "published_at": it.published_at or ""}],
        ))
    brief = f"本板块抓取到 {len(raw_items)} 条资讯；LLM 摘要不可用时采用标题级回退展示。"
    return brief, events

# -------------------------
# Digest build / render
# -------------------------

def normalize_sections_cfg(cfg_sections) -> list[dict]:
    # Support both list and dict
    if isinstance(cfg_sections, list):
        return cfg_sections
    if isinstance(cfg_sections, dict):
        out = []
        for sid, sc in cfg_sections.items():
            if isinstance(sc, dict):
                sc = dict(sc)
                sc.setdefault("id", sid)
                out.append(sc)
        return out
    raise ValueError("config.sections must be list or dict")

def build_digest(cfg: dict, *, date: str, model: str, limit_raw: int, items_per_section: int) -> dict:
    # KPIs
    kpis: list[dict] = []
    kpi_cfg = cfg.get("kpis") or []
    log(f"[digest] kpi start: {len(kpi_cfg)}")
    
for k in kpi_cfg:
        try:
            series = (k.get("series") or "").strip()
            title = (k.get("title") or k.get("name") or series or "KPI").strip()
            unit = (k.get("unit") or "").strip()
            lookback = int(k.get("lookback") or 40)

            values: list[float] = []
            dates: list[str] = []
            if series:
                values, dates = fetch_fred_recent(series, max_points=lookback)

            val = values[-1] if values else None
            updated = dates[-1] if dates else None

            delta = None
            delta_pct = None
            direction = None
            if values and len(values) >= 2 and val is not None:
                prev = values[-2]
                delta = val - prev
                if prev != 0:
                    delta_pct = (delta / prev) * 100.0
                # direction used for UI chips
                if abs(delta) < 1e-12:
                    direction = "flat"
                elif delta > 0:
                    direction = "up"
                else:
                    direction = "down"

            color = _color_for_key(series or title)
            spark = _spark_svg(values[-min(len(values), 40):], stroke=color) if values else ""

            kpis.append({
                "id": series or title,
                "name": title,
                "unit": unit,
                "value": val,
                "updated_at": updated,
                "source": "FRED" if series else (k.get("source") or ""),
                "delta": delta,
                "delta_pct": delta_pct,
                "direction": direction,
                "spark_svg": spark,
                "color": color,
                "series_values": values[-min(len(values), 40):] if values else [],
            })
        
            except Exception as e:
            series = (k.get("series") or "").strip()
            title = (k.get("title") or k.get("name") or series or "KPI").strip()
            unit = (k.get("unit") or "").strip()
            color = _color_for_key(series or title)
            kpis.append({
                "id": series or title,
                "name": title,
                "unit": unit,
                "value": None,
                "updated_at": None,
                "source": "FRED" if series else (k.get("source") or ""),
                "delta": None,
                "delta_pct": None,
                "direction": None,
                "spark_svg": "",
                "color": color,
                "series_values": [],
                "error": str(e)[:200],
            })

    log(f"[digest] kpi done: {len(kpis)}")

    sections_cfg = normalize_sections_cfg(cfg.get("sections") or [])
    log(f"[digest] sections start: {len(sections_cfg)}")

    sections_out: list[dict] = []
    for sc in sections_cfg:
        sid = sc.get("id")
        name = sc.get("name") or sid
        sec = Section(id=sid, name=name, tags=sc.get("tags") or [])
        log(f"[digest] section start: {sid} ({name})")

        # Source strategy: Google News queries only (stable baseline)
        raw_items: list[RawItem] = []
        for q in (sc.get("gnews_queries") or sc.get("queries") or []):
            try:
                items = fetch_google_news_items(q, limit=limit_raw)
                log(f"[digest] gnews {sid} q='{q[:24]}...' entries={len(items)}")
                raw_items.extend(items)
            except Exception as e:
                log(f"[digest] gnews {sid} failed: {type(e).__name__}: {e}")

        raw_items = dedupe_items(raw_items)
        # Keep only most recent-ish first (Google already mostly recent)
        raw_items = raw_items[: max(items_per_section * 3, limit_raw)]

        # Build section pack
        brief_zh = ""
        events: list[Event] = []
        if raw_items:
            try:
                brief_zh, events = llm_section_pack(model, sec, raw_items, items_per_section=items_per_section)
            except Exception as e:
                log(f"[digest] llm pack failed ({sid}): {type(e).__name__}: {e}")
                brief_zh, events = fallback_pack(sec, raw_items, items_per_section=items_per_section)
        else:
            brief_zh = "（今日该板块暂无可用内容）"
            events = []

        # Ensure section is never dropped, so the page won't be blank.
        sections_out.append({
            "id": sec.id,
            "name": sec.name,
            "tags": sec.tags,
            "brief_zh": brief_zh,
            "brief_cn": brief_zh,  # template expects brief_cn sometimes
            "brief_us": brief_zh,
            "brief_eu": brief_zh,
            "brief_asia": brief_zh,
            "brief_global": brief_zh,
            "events": [dataclasses.asdict(e) for e in events],
        })
        log(f"[digest] section done: {sid} events={len(events)} raw={len(raw_items)}")

    log(f"[digest] sections done: {len(sections_out)}")

    digest = {
        "date": date,
        "generated_at_utc": now_utc().isoformat(),
        "kpis": kpis,
        "sections": sections_out,  # IMPORTANT: list (template uses forEach)
    }
    return digest

def render_html(template_path: str, digest: dict) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")
    t = Template(tpl)
    context = {
        "DATE": digest.get("date"),
        "TITLE": f"Ben Daily Digest · {digest.get('date')}",
        "DIGEST_JSON": safe_json_dumps(digest),
    }
    return t.render(**context)

def install_signal_handlers():
    # Make watchdog SIGUSR1 non-fatal and dump stacks
    faulthandler.enable(all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        log("[digest] faulthandler registered on SIGUSR1")
    except Exception:
        pass

# -------------------------
# CLI
# -------------------------

def main():
    install_signal_handlers()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items_per_section", type=int, default=15)
    args = ap.parse_args()

    log("[digest] boot")
    date = args.date or now_utc().date().isoformat()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    log(f"[digest] date={date} model={args.model} limit_raw={args.limit_raw} items={args.items_per_section}")

    digest = build_digest(cfg, date=date, model=args.model, limit_raw=args.limit_raw, items_per_section=args.items_per_section)
    log("[digest] render html")
    html_out = render_html(args.template, digest)
    Path(args.out).write_text(html_out, encoding="utf-8")
    log(f"[digest] wrote: {args.out}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log("[digest] fatal error")
        traceback.print_exc()
        sys.exit(1)
