#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py (stable, low-dependency)
- No feedparser / jinja2
- Template insertion via string replacement of "__DIGEST_JSON__"
- Google News RSS + optional Atom/RSS feeds
- LLM packing per-section (1 call/section) + optional daily brief (1 call)
- SIGUSR1 stack dump support via faulthandler.register()

Expected template:
- daily_digest_template_v15_apple.html contains literal placeholder "__DIGEST_JSON__"
- page JS expects:
  window.DIGEST = { date, generated_at_bjt, sections:[{id,name,events,summary_zh}], kpis:{items:[]}, config:{...} }
"""

from __future__ import annotations

import argparse
import datetime as dt
import faulthandler
import html
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------------
# Utilities
# -------------------------

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def to_bjt(ts: dt.datetime) -> dt.datetime:
    return ts.astimezone(dt.timezone(dt.timedelta(hours=8)))

def iso_date(d: dt.date) -> str:
    return d.isoformat()

def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return default if v is None else v

def retry(fn, *, tries: int = 3, base_sleep: float = 1.0, max_sleep: float = 6.0, jitter: float = 0.2, what: str = "op"):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep = min(max_sleep, base_sleep * (2 ** i))
            sleep = sleep * (1.0 + random.uniform(-jitter, jitter))
            print(f"[digest] retry {what}: {type(e).__name__}: {e} (sleep {sleep:.1f}s)", flush=True)
            time.sleep(max(0.1, sleep))
    raise last  # type: ignore[misc]

def http_get(url: str, *, timeout: float = 20.0, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    h = {"User-Agent": "ben-daily-digest/15 (+https://github.com/)"}  # keep simple
    if headers:
        h.update(headers)
    return requests.get(url, headers=h, timeout=timeout)

def http_post(url: str, *, json_body: Dict[str, Any], timeout: float = 60.0, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    h = {"User-Agent": "ben-daily-digest/15 (+https://github.com/)"}
    if headers:
        h.update(headers)
    return requests.post(url, headers=h, json=json_body, timeout=timeout)


# -------------------------
# Data models
# -------------------------

@dataclass
class RawItem:
    title: str
    url: str
    source: str
    published: Optional[str] = None
    snippet: str = ""

@dataclass
class EventCard:
    id: str
    title: str
    title_zh: str
    source_hint: str
    date: str
    region: str   # "china" | "global" | "mix"
    summary_zh: str
    url: str


# -------------------------
# Feed ingestion
# -------------------------

def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return _strip("".join(el.itertext()))

def parse_rss_or_atom(xml_bytes: bytes, *, default_source: str) -> List[RawItem]:
    """Parse basic RSS2 or Atom feed."""
    items: List[RawItem] = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return items

    # RSS
    channel = root.find("./channel")
    if channel is not None:
        for it in channel.findall("./item"):
            title = _text(it.find("title"))
            link = _text(it.find("link"))
            pub = _text(it.find("pubDate"))
            src = _text(it.find("source")) or default_source
            desc = _text(it.find("description"))
            if title and link:
                items.append(RawItem(title=title, url=link, source=src, published=pub, snippet=desc))
        return items

    # Atom
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//a:entry", ns):
        title = _text(entry.find("a:title", ns))
        # prefer rel="alternate"
        link = ""
        for l in entry.findall("a:link", ns):
            rel = l.attrib.get("rel", "alternate")
            if rel == "alternate" and l.attrib.get("href"):
                link = l.attrib["href"]
                break
        if not link:
            l0 = entry.find("a:link", ns)
            if l0 is not None and l0.attrib.get("href"):
                link = l0.attrib["href"]
        updated = _text(entry.find("a:updated", ns))
        src = default_source
        summ = _text(entry.find("a:summary", ns)) or _text(entry.find("a:content", ns))
        if title and link:
            items.append(RawItem(title=title, url=link, source=src, published=updated, snippet=summ))
    return items

def google_news_rss(query: str, *, lang: str = "en", region: str = "US", max_items: int = 60) -> List[RawItem]:
    params = {
        "q": query,
        "hl": lang,
        "gl": region,
        "ceid": f"{region}:{lang}",
    }
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)
    resp = retry(lambda: http_get(url, timeout=25.0), what="gnews")
    resp.raise_for_status()
    items = parse_rss_or_atom(resp.content, default_source="Google News")
    # google titles are like "Title - Source"
    out: List[RawItem] = []
    for it in items[:max_items]:
        title = it.title
        src = it.source
        if " - " in title:
            t, s = title.rsplit(" - ", 1)
            title = t.strip()
            src = s.strip() or src
        out.append(RawItem(title=title, url=it.url, source=src, published=it.published, snippet=it.snippet))
    return out

def fetch_feed(url: str, *, source: str) -> List[RawItem]:
    resp = retry(lambda: http_get(url, timeout=25.0), what=f"feed {source}")
    resp.raise_for_status()
    return parse_rss_or_atom(resp.content, default_source=source)

def normalize_title(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t)
    return _strip(t)

def dedup_items(items: List[RawItem], *, max_out: int) -> List[RawItem]:
    seen = set()
    out = []
    for it in items:
        k = normalize_title(it.title)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
        if len(out) >= max_out:
            break
    return out


# -------------------------
# KPIs (FRED graph CSV, no API key)
# -------------------------

@dataclass
class KPI:
    id: str
    name: str
    value: Optional[float]
    unit: str
    delta: Optional[float]
    source: str

def fred_latest(series_id: str) -> Tuple[Optional[float], Optional[float]]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(series_id)}"
    resp = retry(lambda: http_get(url, timeout=25.0), what=f"fred {series_id}", tries=4)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    if len(lines) < 3:
        return None, None
    # header: DATE,VALUE
    vals = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) != 2:
            continue
        v = parts[1].strip()
        if v in (".", ""):
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return None, None
    last = vals[-1]
    prev = vals[-2] if len(vals) >= 2 else None
    delta = (last - prev) if (prev is not None) else None
    return last, delta

def build_kpis(cfg: Dict[str, Any]) -> List[KPI]:
    kpis_cfg = cfg.get("kpis", []) or []
    out: List[KPI] = []
    for k in kpis_cfg:
        sid = str(k.get("id", "")).strip()
        if not sid:
            continue
        name = str(k.get("name", sid))
        unit = str(k.get("unit", ""))
        src = str(k.get("source", "FRED"))
        try:
            v, d = fred_latest(sid)
        except Exception as e:
            print(f"[digest] kpi failed {sid}: {type(e).__name__}: {e}", flush=True)
            v, d = None, None
        out.append(KPI(id=sid, name=name, value=v, unit=unit, delta=d, source=src))
    return out


# -------------------------
# OpenAI (Chat Completions)
# -------------------------

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

def openai_chat_json(*, model: str, messages: List[Dict[str, str]], max_tokens: int = 900, temperature: float = 0.2) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    def _do():
        r = http_post(OPENAI_CHAT_URL, json_body=body, timeout=75.0, headers=headers)
        # Keep raw for debugging if 400/429
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.text[:300]}")
        return r.json()

    data = retry(_do, what="openai", tries=3, base_sleep=1.0, max_sleep=8.0)
    content = data["choices"][0]["message"]["content"]
    # Strict JSON parse; if model adds prose, attempt to extract first JSON object
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            raise ValueError("OpenAI did not return JSON object")
        return json.loads(m.group(0))


def llm_section_pack(*, model: str, section_id: str, section_name: str, raw_items: List[RawItem], items_per_section: int) -> Tuple[str, List[EventCard]]:
    # Build compact candidate list
    candidates = []
    for it in raw_items[:min(len(raw_items), 30)]:
        candidates.append({
            "title": it.title[:160],
            "source": it.source[:80],
            "published": (it.published or "")[:64],
            "url": it.url,
            "snippet": (it.snippet or "")[:240],
        })

    sys_prompt = (
        "You are a bilingual (EN->ZH) news analyst.\n"
        "Task: From candidate news items, cluster duplicates, prioritize the most consequential events, "
        "and output concise 'conclusion-first' event cards in Simplified Chinese.\n"
        "Rules:\n"
        f"- Output STRICT JSON ONLY.\n"
        f"- Produce up to {items_per_section} cards.\n"
        "- Each card must be a single event cluster, not a single article.\n"
        "- Each card.summary_zh MUST start with '结论：' then '依据：' then '影响：'.\n"
        "- region must be one of: china, global, mix.\n"
        "- Choose ONE best url from candidates as the card url.\n"
    )

    user_payload = {
        "section_id": section_id,
        "section_name": section_name,
        "candidates": candidates,
    }

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    out = openai_chat_json(model=model, messages=messages, max_tokens=1100, temperature=0.2)
    summary_zh = _strip(str(out.get("summary_zh", "")))[:1200]
    cards_raw = out.get("cards", []) or []
    cards: List[EventCard] = []
    for idx, c in enumerate(cards_raw[:items_per_section]):
        try:
            card = EventCard(
                id=str(c.get("id") or f"{section_id}-{idx+1}"),
                title=str(c.get("title", ""))[:180],
                title_zh=str(c.get("title_zh", ""))[:180],
                source_hint=str(c.get("source_hint", ""))[:120],
                date=str(c.get("date", ""))[:32],
                region=str(c.get("region", "global"))[:16],
                summary_zh=_strip(str(c.get("summary_zh", "")))[:1800],
                url=str(c.get("url", ""))[:500],
            )
            if not card.url:
                continue
            cards.append(card)
        except Exception:
            continue

    # If LLM returns nothing, fallback: take top raw items
    if not cards:
        for it in raw_items[:min(items_per_section, len(raw_items))]:
            cards.append(EventCard(
                id=f"{section_id}-{len(cards)+1}",
                title=it.title[:180],
                title_zh=it.title[:180],
                source_hint=it.source[:120],
                date=(it.published or "")[:32],
                region="global",
                summary_zh=f"结论：{it.title}\n依据：{_strip(it.snippet)[:180]}\n影响：待观察。",
                url=it.url,
            ))
        if not summary_zh:
            summary_zh = f"{section_name}：今日无明显高置信度聚类事件，已按信息密度输出候选条目。"

    if not summary_zh:
        summary_zh = f"{section_name}：基于候选新闻聚类后的要点总结。"

    return summary_zh, cards


def llm_daily_brief(*, model: str, sections: List[Dict[str, Any]], kpis: List[KPI]) -> Tuple[str, List[EventCard]]:
    # compress input
    sec_summaries = []
    for s in sections:
        sec_summaries.append({
            "id": s["id"],
            "name": s["name"],
            "summary_zh": (s.get("summary_zh") or "")[:500],
            "top_titles": [e["title_zh"] for e in s.get("events", [])[:5]],
        })
    kpi_lines = []
    for k in kpis:
        if k.value is None:
            continue
        delta_str = f"{k.delta:.3f}" if (k.delta is not None) else "NA"
        kpi_lines.append(f"{k.name}: {k.value:.3f}{k.unit} (Δ {delta_str})")

    sys_prompt = (
        "You are a Chinese news editor.\n"
        "Task: Write a concise daily top brief for an executive. "
        "Output STRICT JSON ONLY.\n"
        "Rules:\n"
        "- Produce 5~10 top points as event cards (same schema as section cards).\n"
        "- Each summary_zh must start with '结论：' then '依据：' then '影响：'.\n"
        "- Prefer cross-section synthesis and highlight risk/impact.\n"
    )
    user_payload = {"kpis": kpi_lines[:10], "sections": sec_summaries}

    out = openai_chat_json(model=model, messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ], max_tokens=900, temperature=0.2)

    summary_zh = _strip(str(out.get("summary_zh", "")))[:1400]
    cards: List[EventCard] = []
    for idx, c in enumerate((out.get("cards", []) or [])[:10]):
        url = str(c.get("url", "")).strip()
        if not url:
            continue
        cards.append(EventCard(
            id=str(c.get("id") or f"top-{idx+1}"),
            title=str(c.get("title", ""))[:180],
            title_zh=str(c.get("title_zh", ""))[:180],
            source_hint=str(c.get("source_hint", ""))[:120],
            date=str(c.get("date", ""))[:32],
            region=str(c.get("region", "mix"))[:16],
            summary_zh=_strip(str(c.get("summary_zh", "")))[:1800],
            url=url[:500],
        ))
    if not summary_zh:
        summary_zh = "Top 今日要点：基于各栏目事件聚类与关键指标的综合摘要。"
    if not cards:
        # fallback: take first event of each section
        for s in sections:
            evs = s.get("events", [])
            if evs:
                e = evs[0]
                cards.append(EventCard(
                    id=f"top-{len(cards)+1}",
                    title=e["title"],
                    title_zh=e["title_zh"],
                    source_hint=e["source_hint"],
                    date=e["date"],
                    region=e.get("region","mix"),
                    summary_zh=e["summary_zh"],
                    url=e["url"],
                ))
        cards = cards[:8]
    return summary_zh, cards


# -------------------------
# Config compatibility
# -------------------------

def load_config(path: str) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg if isinstance(cfg, dict) else {"sections": cfg}

def normalize_sections_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accepts:
    - cfg["sections"] as list[section]
    - cfg["sections"] as dict[id->section]
    """
    sc = cfg.get("sections", [])
    out: List[Dict[str, Any]] = []
    if isinstance(sc, dict):
        for sid, sv in sc.items():
            if isinstance(sv, dict):
                sv = dict(sv)
                sv.setdefault("id", sid)
                out.append(sv)
    elif isinstance(sc, list):
        for sv in sc:
            if isinstance(sv, dict):
                out.append(sv)
    # stable order by "order" then id
    out.sort(key=lambda x: (int(x.get("order", 9999)), str(x.get("id", ""))))
    return out


# -------------------------
# Digest assembly
# -------------------------

def build_raw_items_for_section(sec: Dict[str, Any], *, limit_raw: int) -> List[RawItem]:
    items: List[RawItem] = []

    # Feeds
    for f in (sec.get("feeds") or []):
        if not isinstance(f, dict):
            continue
        url = str(f.get("url", "")).strip()
        if not url:
            continue
        src = str(f.get("source", "Feed")).strip() or "Feed"
        try:
            its = fetch_feed(url, source=src)
            items.extend(its)
        except Exception as e:
            print(f"[digest] feed {sec.get('id')} failed: {src}: {type(e).__name__}: {e}", flush=True)

    # Google News queries
    for q in (sec.get("gnews_queries") or []):
        if not isinstance(q, str) or not q.strip():
            continue
        try:
            its = google_news_rss(q.strip(), lang="en", region="US", max_items=80)
            print(f"[digest] gnews {sec.get('id')} q='{q[:28]}...' entries={len(its)}", flush=True)
            items.extend(its)
        except Exception as e:
            print(f"[digest] gnews {sec.get('id')} failed: {type(e).__name__}: {e}", flush=True)

    # basic cleanup + dedup
    items = [it for it in items if it.title and it.url]
    # cheap: keep "published" missing ok
    items = dedup_items(items, max_out=limit_raw)
    return items

def build_digest(cfg: Dict[str, Any], *, date: str, model: str, limit_raw: int, items_per_section: int) -> Dict[str, Any]:
    print("[digest] kpi start:", len(cfg.get("kpis", []) or []), flush=True)
    kpis = build_kpis(cfg)
    print("[digest] kpi done:", len([k for k in kpis if k.value is not None]), flush=True)

    sections_cfg = normalize_sections_cfg(cfg)
    print("[digest] sections start:", len(sections_cfg), flush=True)

    sections_out: List[Dict[str, Any]] = []

    for sec in sections_cfg:
        sid = str(sec.get("id", "")).strip()
        name = str(sec.get("name", sid)).strip()
        if not sid:
            continue
        print(f"[digest] section start: {sid} ({name})", flush=True)

        raw_items = build_raw_items_for_section(sec, limit_raw=limit_raw)

        # LLM pack for the section (skip if no candidates)
        if not raw_items:
            summary_zh = f"{name}：今日未检索到足够的候选新闻（或源不可用）。"
            cards: List[EventCard] = []
        else:
            try:
                summary_zh, cards = llm_section_pack(model=model, section_id=sid, section_name=name, raw_items=raw_items, items_per_section=items_per_section)
            except Exception as e:
                print(f"[digest] llm pack failed {sid}: {type(e).__name__}: {e}", flush=True)
                summary_zh = f"{name}：LLM 聚类失败，已输出候选条目（降级）。"
                cards = [
                    EventCard(
                        id=f"{sid}-{i+1}",
                        title=it.title[:180],
                        title_zh=it.title[:180],
                        source_hint=it.source[:120],
                        date=(it.published or "")[:32],
                        region="global",
                        summary_zh=f"结论：{it.title}\n依据：{_strip(it.snippet)[:180]}\n影响：待观察。",
                        url=it.url,
                    )
                    for i, it in enumerate(raw_items[:min(items_per_section, len(raw_items))])
                ]

        sections_out.append({
            "id": sid,
            "name": name,
            "summary_zh": summary_zh,
            "events": [card.__dict__ for card in cards],
        })
        print(f"[digest] section done: {sid} events={len(cards)}", flush=True)

    # Optional daily top brief (as first pseudo-section)
    try:
        top_summary, top_cards = llm_daily_brief(model=model, sections=sections_out, kpis=kpis)
        top_section = {
            "id": "top",
            "name": "Top 今日要点",
            "summary_zh": top_summary,
            "events": [c.__dict__ for c in top_cards],
        }
        sections_out = [top_section] + sections_out
    except Exception as e:
        print(f"[digest] llm daily brief failed: {type(e).__name__}: {e}", flush=True)
        # keep sections_out unchanged

    digest = {
        "date": date,
        "generated_at_bjt": to_bjt(now_utc()).strftime("%Y-%m-%d %H:%M:%S"),
        "kpis": {
            "items": [k.__dict__ for k in kpis],
        },
        "sections": sections_out,
        "config": {
            "items_per_section": items_per_section,
            "limit_raw": limit_raw,
            "model": model,
        },
    }
    return digest


def render_html(template_path: str, digest: Dict[str, Any]) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")
    payload = json.dumps(digest, ensure_ascii=False)
    if "__DIGEST_JSON__" not in tpl:
        raise RuntimeError("Template missing __DIGEST_JSON__ placeholder")
    return tpl.replace("__DIGEST_JSON__", payload)


# -------------------------
# CLI
# -------------------------

from pathlib import Path  # intentionally here (used above)

def main() -> None:
    # enable SIGUSR1 stack dump without killing the process
    try:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default="")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items_per_section", type=int, default=15)
    args = ap.parse_args()

    # date default: BJT "today"
    if args.date:
        date = args.date
    else:
        date = iso_date(to_bjt(now_utc()).date())

    print("[digest] boot", flush=True)
    print(f"[digest] date={date} model={args.model} limit_raw={args.limit_raw} items={args.items_per_section}", flush=True)

    cfg = load_config(args.config)
    digest = build_digest(cfg, date=date, model=args.model, limit_raw=args.limit_raw, items_per_section=args.items_per_section)

    out_html = render_html(args.template, digest)
    Path(args.out).write_text(out_html, encoding="utf-8")
    print(f"[digest] wrote: {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("[digest] fatal:", type(e).__name__, str(e), flush=True)
        traceback.print_exc()
        sys.exit(1)
