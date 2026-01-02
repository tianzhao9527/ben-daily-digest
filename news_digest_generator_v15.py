#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""news_digest_generator_v15.py

Fixes for v15/v18 template mismatch and GH Actions watchdog:
- Proper server-side Jinja2 rendering when template contains Jinja markers ({% ... %}, {{ ... }}).
- Fallback to JSON injection only when template contains __DIGEST_JSON__.
- SIGUSR1 stack dump (faulthandler) so watchdog can dump without killing the process.
- RSS feed config can be dict {source,url} or plain string URL.
- All HTTP calls have timeouts + bounded retries to prevent hanging.
- LLM JSON parsing is robust; falls back to deterministic brief when LLM fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import faulthandler
import hashlib
import html as _html
import json
import math
import os
import random
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import requests

try:
    from jinja2 import Environment, BaseLoader, StrictUndefined
except Exception:
    Environment = None  # type: ignore


DEFAULT_UA = "ben-daily-digest/1.0 (+github-actions)"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def log(msg: str) -> None:
    print(msg, flush=True)


def now_beijing() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(
        dt.timezone(dt.timedelta(hours=8))
    )


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def retry(fn, *, tries: int = 3, base_sleep: float = 1.0, jitter: float = 0.4, what: str = "op"):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep_s = base_sleep * (1.0 + i * 0.6) * (1.0 + random.random() * jitter)
            log(f"[digest] retry {what}: {type(e).__name__}: {e} (sleep {sleep_s:.1f}s)")
            time.sleep(sleep_s)
    raise last  # type: ignore


def http_get(url: str, *, timeout_s: float = 20.0, headers: Optional[dict] = None) -> requests.Response:
    hdr = {"User-Agent": DEFAULT_UA, "Accept": "*/*"}
    if headers:
        hdr.update(headers)
    s = requests.Session()
    return s.get(url, headers=hdr, timeout=(6.0, timeout_s))


def http_post(url: str, *, json_body: dict, timeout_s: float = 60.0, headers: Optional[dict] = None) -> requests.Response:
    hdr = {"User-Agent": DEFAULT_UA, "Accept": "application/json", "Content-Type": "application/json"}
    if headers:
        hdr.update(headers)
    s = requests.Session()
    return s.post(url, headers=hdr, json=json_body, timeout=(10.0, timeout_s))


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    obj = safe_json_loads(text.strip())
    if isinstance(obj, dict):
        return obj
    # find first {...} that parses
    for m in re.finditer(r"\{", text):
        start = m.start()
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            chunk = text[start:end]
            obj = safe_json_loads(chunk)
            if isinstance(obj, dict):
                return obj
    return None


@dataclass
class RawItem:
    title: str
    url: str
    source: str
    published: str
    snippet: str
    section_id: str


@dataclass
class EventCard:
    title: str
    one_liner: str
    key_points: List[str]
    why_it_matters: List[str]
    action: List[str]
    sources: List[dict]
    tags: List[str]


@dataclass
class KPI:
    id: str
    title: str
    unit: str
    value: Optional[float]
    delta: Optional[float]
    direction: Optional[str]
    source: str
    spark: str


def google_news_rss_url(query: str, *, lang: str = "en", gl: str = "US", ceid: str = "US:en") -> str:
    from urllib.parse import quote_plus
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={lang}&gl={gl}&ceid={ceid}"


def fetch_google_news_items(query: str, *, section_id: str, max_entries: int = 50) -> List[RawItem]:
    url = google_news_rss_url(query)

    def _do():
        r = http_get(url, timeout_s=25.0)
        r.raise_for_status()
        return r.text

    xml = retry(_do, tries=3, what=f"gnews {section_id}")
    feed = feedparser.parse(xml)
    items: List[RawItem] = []
    for e in feed.entries[:max_entries]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        source = "Google News"
        if " - " in title:
            t, s = title.rsplit(" - ", 1)
            if len(s) <= 50:
                title = t.strip()
                source = s.strip()
        summary = getattr(e, "summary", "") or ""
        summary = re.sub(r"<[^>]+>", "", summary).strip()
        if title and link:
            items.append(RawItem(title=title, url=link, source=source, published=published, snippet=summary, section_id=section_id))
    return items


def parse_rss_items(feed: Any, *, section_id: str, max_entries: int = 50) -> List[RawItem]:
    if isinstance(feed, dict):
        source_name = str(feed.get("source", "") or "RSS")
        url = str(feed.get("url", "") or "").strip()
    else:
        source_name = "RSS"
        url = str(feed or "").strip()

    if not url:
        return []

    def _do():
        r = http_get(url, timeout_s=25.0)
        r.raise_for_status()
        return r.text

    try:
        xml = retry(_do, tries=2, what=f"feed {source_name}")
    except Exception as e:
        log(f"[digest] feed {section_id} failed: {e}")
        return []

    parsed = feedparser.parse(xml)
    items: List[RawItem] = []
    for e in parsed.entries[:max_entries]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        summary = getattr(e, "summary", "") or ""
        summary = re.sub(r"<[^>]+>", "", summary).strip()
        if title and link:
            items.append(RawItem(title=title, url=link, source=source_name, published=published, snippet=summary, section_id=section_id))
    return items


def fetch_fred_series_csv(series_id: str) -> List[Tuple[str, float]]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    def _do():
        r = http_get(url, timeout_s=25.0)
        r.raise_for_status()
        return r.text

    csv_text = retry(_do, tries=3, what=f"fred {series_id}")
    rows: List[Tuple[str, float]] = []
    for line in csv_text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        ds, vs = parts
        if vs in (".", ""):
            continue
        try:
            rows.append((ds, float(vs)))
        except Exception:
            continue
    return rows


def sparkline_svg(values: List[float], *, width: int = 120, height: int = 28, pad: int = 2) -> str:
    if not values:
        return ""
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = pad + (width - 2 * pad) * (i / (n - 1 if n > 1 else 1))
        y = pad + (height - 2 * pad) * (1.0 - (v - vmin) / (vmax - vmin))
        pts.append((x, y))
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<path d="{d}" fill="none" stroke="currentColor" stroke-width="2" />'
        f'</svg>'
    )


def build_kpis(cfg: dict) -> List[KPI]:
    out: List[KPI] = []
    for k in cfg.get("kpis", []) or []:
        kid = k.get("id", "")
        title = k.get("title", kid)
        unit = k.get("unit", "")
        source = k.get("source", "FRED")
        series = k.get("fred_series")
        value = None
        delta = None
        direction = None
        spark = ""
        if series:
            rows = fetch_fred_series_csv(series)
            vals = [v for _, v in rows][-30:]
            if vals:
                value = vals[-1]
                if len(vals) >= 2:
                    delta = vals[-1] - vals[-2]
                    direction = "up" if delta > 0 else ("down" if delta < 0 else None)
                spark = sparkline_svg(vals)
        out.append(KPI(id=kid, title=title, unit=unit, value=value, delta=delta, direction=direction, source=source, spark=spark))
    return out


def openai_chat(model: str, messages: List[dict], *, temperature: float = 0.2, max_tokens: int = 900, timeout_s: float = 90.0) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = http_post(OPENAI_URL, json_body=payload, timeout_s=timeout_s, headers={"Authorization": f"Bearer {api_key}"})
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code} {r.text[:400]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def openai_chat_json(model: str, messages: List[dict], *, temperature: float = 0.2, max_tokens: int = 900) -> dict:
    txt = retry(lambda: openai_chat(model, messages, temperature=temperature, max_tokens=max_tokens), tries=3, what="openai")
    obj = extract_first_json_object(txt)
    if not isinstance(obj, dict):
        raise ValueError("OpenAI did not return JSON object")
    return obj


def dedupe_items(items: List[RawItem]) -> List[RawItem]:
    seen = set()
    out = []
    for it in items:
        k = sha1((it.title + "|" + it.url).lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def pick_candidates(items: List[RawItem], *, limit_raw: int) -> List[RawItem]:
    items = sorted(items, key=lambda x: (len(x.snippet or ""), x.source != "Google News"), reverse=True)
    return items[:limit_raw]


def llm_section_pack(model: str, section_name: str, raw_items: List[RawItem], *, items: int = 15) -> List[EventCard]:
    rows = []
    for i, it in enumerate(raw_items[:40], 1):
        rows.append({"i": i, "title": it.title, "source": it.source, "published": it.published, "url": it.url, "snippet": (it.snippet or "")[:240]})

    sys_prompt = "You are an analyst generating a daily briefing. Return ONLY a JSON object. No markdown."
    user_prompt = {
        "section": section_name,
        "items": rows,
        "output_spec": {
            "events": [{
                "title": "string (<=40 chars, Chinese)",
                "one_liner": "string (<=60 chars, Chinese)",
                "key_points": ["..."],
                "why_it_matters": ["..."],
                "action": ["..."],
                "sources": [{"i": 1}],
                "tags": ["..."],
            }]
        },
        "rules": [
            f"Output up to {items} events; if low-signal, output fewer but not empty.",
            "Each event must cite 1-3 sources by index i from input items.",
            "Be concrete: numbers, names, decisions, dates.",
        ],
    }

    obj = openai_chat_json(model, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}], temperature=0.2, max_tokens=1200)
    events = obj.get("events") or obj.get("Events") or []
    out: List[EventCard] = []
    for e in events:
        try:
            sources = []
            for s in (e.get("sources") or [])[:3]:
                idx = int(s.get("i", 0))
                if 1 <= idx <= len(rows):
                    ri = rows[idx - 1]
                    sources.append({"title": ri["title"], "url": ri["url"], "source": ri["source"], "published": ri["published"]})
            out.append(EventCard(
                title=str(e.get("title", "")).strip(),
                one_liner=str(e.get("one_liner", "")).strip(),
                key_points=[str(x).strip() for x in (e.get("key_points") or []) if str(x).strip()],
                why_it_matters=[str(x).strip() for x in (e.get("why_it_matters") or []) if str(x).strip()],
                action=[str(x).strip() for x in (e.get("action") or []) if str(x).strip()],
                sources=sources,
                tags=[str(x).strip() for x in (e.get("tags") or []) if str(x).strip()][:6],
            ))
        except Exception:
            continue
    return [x for x in out if x.title and x.sources]


def fallback_event_cards(section_name: str, items: List[RawItem], *, max_events: int = 6) -> List[EventCard]:
    out: List[EventCard] = []
    for it in items[:max_events]:
        out.append(EventCard(
            title=it.title[:48],
            one_liner=(it.snippet[:60] if it.snippet else it.title[:60]),
            key_points=[(it.snippet[:140] if it.snippet else it.title)[:140]],
            why_it_matters=[f"{section_name}：建议跟踪该信息的后续进展与量化影响。"],
            action=["打开原文核对口径与时间点；如涉及合规/制裁，确认适用范围与对象。"],
            sources=[{"title": it.title, "url": it.url, "source": it.source, "published": it.published}],
            tags=[section_name.split("（")[0][:6]],
        ))
    return out


def build_top_brief_fallback(sections: List[dict], kpis: List[KPI]) -> dict:
    bullets = []
    for sec in sections:
        for ev in sec.get("events", [])[:2]:
            bullets.append(f"{sec.get('name','')}: {ev.one_liner or ev.title}")
            if len(bullets) >= 5:
                break
        if len(bullets) >= 5:
            break

    kpi_bullets = []
    for k in kpis[:6]:
        if k.value is None:
            continue
        dv = ""
        if k.delta is not None:
            sign = "+" if k.delta > 0 else ""
            dv = f" (Δ {sign}{k.delta:.3f})"
        kpi_bullets.append(f"{k.title}: {k.value:.3f}{k.unit}{dv}")

    return {
        "one_liner": "今日信息偏结构化风险与供需变化，建议以‘结论优先’方式跟踪关键变量。",
        "key_changes": bullets or ["暂无足够高置信事件（或 LLM 输出异常），建议以数据与公告为主线跟踪。"],
        "actions": ["优先核对 2-3 条高影响新闻原文与数据口径；", "把关键阈值写入决策面板，次日自动对比变化。"],
        "kpis": kpi_bullets,
    }


def split_views(sections: List[dict]) -> Dict[str, dict]:
    def is_cn(ev: EventCard) -> bool:
        txt = " ".join([ev.title, ev.one_liner] + ev.key_points + ev.why_it_matters)
        return any(x in txt for x in ["中国", "China", "CNY", "人民币", "大陆", "香港", "台湾"])

    views = {"all": {"sections": sections}, "cn": {"sections": []}, "global": {"sections": []}}
    for sec in sections:
        evs: List[EventCard] = sec.get("events", [])
        cn = [e for e in evs if is_cn(e)]
        gl = [e for e in evs if not is_cn(e)]
        if cn:
            views["cn"]["sections"].append({**sec, "events": cn})
        if gl:
            views["global"]["sections"].append({**sec, "events": gl})
    return views


def render_html(template_path: str, context: dict) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")

    # Jinja2 rendering (recommended)
    if Environment is not None and ("{%" in tpl or "{{" in tpl):
        env = Environment(loader=BaseLoader(), autoescape=True, undefined=StrictUndefined)
        t = env.from_string(tpl)
        return t.render(**context)

    # JSON injection fallback
    digest_json = json.dumps(context, ensure_ascii=False)
    if "__DIGEST_JSON__" in tpl:
        return tpl.replace("__DIGEST_JSON__", _html.escape(digest_json))

    return tpl


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_raw_items_for_section(section: dict, *, limit_raw: int) -> List[RawItem]:
    sid = section["id"]
    items: List[RawItem] = []

    for feed in section.get("feeds", []) or []:
        items += parse_rss_items(feed, section_id=sid, max_entries=limit_raw)

    for q in section.get("queries", []) or []:
        q_short = (q[:28] + "...") if len(q) > 31 else q
        g = fetch_google_news_items(q, section_id=sid, max_entries=limit_raw * 4)
        log(f"[digest] gnews {sid} q='{q_short}' entries={len(g)}")
        items += g

    items = dedupe_items(items)
    items = pick_candidates(items, limit_raw=limit_raw)
    return items


def build_digest(cfg: dict, *, date_str: str, model: str, limit_raw: int, items_per_section: int) -> dict:
    log(f"[digest] kpi start: {len(cfg.get('kpis', []))}")
    kpis = build_kpis(cfg)
    log(f"[digest] kpi done: {len(kpis)}")

    sections_out: List[dict] = []
    log(f"[digest] sections start: {len(cfg.get('sections', []))}")
    for sec in cfg.get("sections", []):
        sid = sec["id"]
        name = sec.get("name", sid)
        log(f"[digest] section start: {sid} ({name})")
        raw = build_raw_items_for_section(sec, limit_raw=limit_raw)
        if raw:
            try:
                events = llm_section_pack(model, name, raw, items=items_per_section)
            except Exception as e:
                log(f"[digest] llm section {sid} failed: {e}")
                events = fallback_event_cards(name, raw, max_events=min(6, items_per_section))
        else:
            events = []
        log(f"[digest] section done: {sid} events={len(events)}")
        sections_out.append({"id": sid, "name": name, "events": events})

    log(f"[digest] sections done: {len(sections_out)}")

    # Top brief (LLM optional)
    top_brief = None
    try:
        brief_input = {
            "date": date_str,
            "sections": [{"id": s["id"], "name": s["name"], "event_titles": [e.title for e in s["events"][:5]]} for s in sections_out],
            "kpis": [{"id": k.id, "title": k.title, "unit": k.unit, "value": k.value, "delta": k.delta} for k in kpis],
            "output_spec": {"one_liner": "string", "key_changes": ["..."], "actions": ["..."]},
        }
        obj = openai_chat_json(model, [{"role": "system", "content": "Return ONLY JSON."}, {"role": "user", "content": json.dumps(brief_input, ensure_ascii=False)}], temperature=0.2, max_tokens=600)
        if isinstance(obj, dict) and obj.get("one_liner"):
            top_brief = obj
    except Exception as e:
        log(f"[digest] llm daily brief failed: {e}")

    if not top_brief:
        top_brief = build_top_brief_fallback(sections_out, kpis)

    views = split_views(sections_out)
    views["all"]["top"] = top_brief
    views["cn"]["top"] = top_brief
    views["global"]["top"] = top_brief

    digest = {
        # v15-style uppercase
        "TITLE": cfg.get("title", "Ben 的每日资讯简报"),
        "DATE": date_str,
        "GENERATED_AT": now_beijing().strftime("%Y-%m-%d %H:%M"),
        "SECTIONS": sections_out,
        "KPIS": [dataclasses.asdict(k) for k in kpis],
        "TOP": top_brief,
        # v18-style
        "VIEWS": views,
        "ALERTS": [],
    }
    # lowercase aliases
    digest["title"] = digest["TITLE"]
    digest["date"] = digest["DATE"]
    digest["generated_at"] = digest["GENERATED_AT"]
    digest["sections"] = digest["SECTIONS"]
    digest["kpis"] = digest["KPIS"]
    digest["top"] = digest["TOP"]
    digest["views"] = digest["VIEWS"]
    digest["alerts"] = digest["ALERTS"]
    return digest


def main() -> None:
    # Allow watchdog to send SIGUSR1 without killing the process.
    try:
        faulthandler.register(signal.SIGUSR1)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--date", default="")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items", type=int, default=15)
    args = ap.parse_args()

    log("[digest] boot")
    date_str = args.date.strip() or now_beijing().strftime("%Y-%m-%d")
    log(f"[digest] date={date_str} model={args.model} limit_raw={args.limit_raw} items={args.items}")

    cfg = load_config(args.config)
    digest = build_digest(cfg, date_str=date_str, model=args.model, limit_raw=args.limit_raw, items_per_section=args.items)

    log("[digest] render html")
    out_html = render_html(args.template, digest)
    Path(args.out).write_text(out_html, encoding="utf-8")
    log(f"[digest] wrote: {args.out}")


if __name__ == "__main__":
    main()
