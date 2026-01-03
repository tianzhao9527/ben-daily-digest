#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py
- Fetches raw news items per section (Google News RSS + optional RSS feeds)
- Uses OpenAI to produce concise Chinese briefs per section (optional; robust fallback if LLM fails)
- Renders a single-file HTML using the Apple-style template which expects:
    const D = __DIGEST_JSON__;
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import datetime as _dt
import email.utils
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import requests

import faulthandler
import signal

# Enable stack dumps on demand from GitHub Actions watchdog (kill -USR1 <pid>)
faulthandler.enable(all_threads=True)

def _sigusr1_handler(signum, frame):
    try:
        sys.stderr.write("\n[digest] SIGUSR1 stack dump\n")
        sys.stderr.flush()
    except Exception:
        pass
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        pass

try:
    signal.signal(signal.SIGUSR1, _sigusr1_handler)
except Exception:
    # SIGUSR1 may not exist on some platforms; safe to ignore
    pass



# ----------------------------
# Generic helpers
# ----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def bjt_now() -> _dt.datetime:
    # Beijing time = UTC+8
    return now_utc().astimezone(_dt.timezone(_dt.timedelta(hours=8)))


def parse_rfc2822(dt_str: str) -> Optional[_dt.datetime]:
    if not dt_str:
        return None
    try:
        d = email.utils.parsedate_to_datetime(dt_str)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d
    except Exception:
        return None


def stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def http_get(url: str, timeout: float = 20.0, headers: Optional[Dict[str, str]] = None) -> str:
    # Keep headers minimal to reduce bot detection
    hdr = {
        "User-Agent": "Mozilla/5.0 (compatible; BenDigestBot/1.0; +https://example.invalid)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if headers:
        hdr.update(headers)
    r = requests.get(url, headers=hdr, timeout=timeout)
    r.raise_for_status()
    return r.text


def http_post_json(url: str, payload: Dict[str, Any], timeout: float = 60.0, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    hdr = {"Content-Type": "application/json"}
    if headers:
        hdr.update(headers)
    r = requests.post(url, headers=hdr, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def retry(fn, *, tries: int = 3, base_sleep: float = 1.0, label: str = ""):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep_s = base_sleep * (1.0 + 0.6 * i)
            if label:
                log(f"[digest] retry {label}: {type(e).__name__}: {e} (sleep {sleep_s:.1f}s)")
            time.sleep(sleep_s)
    raise last


# ----------------------------
# Data models
# ----------------------------

@dataclass
class RawItem:
    title: str
    url: str
    source: str
    published: str = ""
    snippet: str = ""
    section: str = ""

    # Compatibility: some older code / templates referred to summary
    @property
    def summary(self) -> str:
        return self.snippet or ""


@dataclass
class KPI:
    id: str
    title: str
    unit: str = ""
    value: Optional[float] = None
    delta: Optional[float] = None
    direction: Optional[str] = None  # "up" / "down"
    delta_pct: Optional[float] = None
    source: str = ""
    spark: str = ""  # HTML fragment


# ----------------------------
# RSS parsing
# ----------------------------

def _strip(s: Optional[str]) -> str:
    return (s or "").strip()


def parse_rss(xml_text: str, *, fallback_source: str = "") -> List[Dict[str, str]]:
    """
    Parses RSS/Atom with ElementTree (no external deps).
    Returns list of dicts with keys: title, link, published, source, summary
    """
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    # Handle RSS 2.0: <rss><channel><item>...
    # Handle Atom: <feed><entry>...
    tag = root.tag.lower()
    if tag.endswith("rss") or "rss" in tag:
        channel = root.find("channel")
        if channel is None:
            channel = root
        for it in channel.findall("item"):
            title = _strip(_find_text(it, "title"))
            link = _strip(_find_text(it, "link"))
            pub = _strip(_find_text(it, "pubDate")) or _strip(_find_text(it, "dc:date"))
            src = _strip(_find_text(it, "source")) or fallback_source
            desc = _strip(_find_text(it, "description"))
            items.append({"title": title, "link": link, "published": pub, "source": src, "summary": desc})
        return items

    # Atom
    # Namespaces: typically {http://www.w3.org/2005/Atom}feed
    for entry in root.findall(".//{*}entry"):
        title = _strip(_find_text(entry, "{*}title"))
        link = ""
        for l in entry.findall("{*}link"):
            href = l.attrib.get("href", "")
            rel = l.attrib.get("rel", "")
            if href and (not rel or rel == "alternate"):
                link = href
                break
        pub = _strip(_find_text(entry, "{*}updated")) or _strip(_find_text(entry, "{*}published"))
        src = fallback_source
        summ = _strip(_find_text(entry, "{*}summary")) or _strip(_find_text(entry, "{*}content"))
        items.append({"title": title, "link": link, "published": pub, "source": src, "summary": summ})
    return items


def _find_text(node: ET.Element, tag: str) -> str:
    # support namespace wildcards via "{*}tag"
    try:
        el = node.find(tag)
        if el is not None and el.text:
            return el.text
        # try without namespace prefix for "dc:date"
        if ":" in tag and not tag.startswith("{"):
            # best-effort: match localname
            local = tag.split(":", 1)[1]
            for child in list(node):
                if child.tag.endswith(local) and child.text:
                    return child.text
    except Exception:
        pass
    return ""


def google_news_rss_url(query: str, *, hl: str, gl: str, ceid: str) -> str:
    q = urllib.parse.quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


# ----------------------------
# OpenAI (JSON mode)
# ----------------------------

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def openai_chat_json(model: str, messages: List[Dict[str, str]], *, timeout: float = 90.0, tries: int = 3) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    def _do():
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            # JSON mode: most modern models accept this.
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        return http_post_json(OPENAI_CHAT_URL, payload, timeout=timeout, headers=headers)

    return retry(_do, tries=tries, base_sleep=1.2, label="openai")


def _extract_json_object(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not s:
        return None
    # Fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Best-effort salvage: take first {...last}
    lb = s.find("{")
    rb = s.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        chunk = s[lb : rb + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def llm_section_pack(model: str, section_id: str, section_title_cn: str, raw_items: List[RawItem], *, items_per_section: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns: (section_brief_zh, events[])
    """
    # Keep prompt compact; we only need zh brief + per-item zh summary.
    raw_payload = []
    for it in raw_items[: min(len(raw_items), 30)]:
        raw_payload.append(
            {
                "title": it.title,
                "source": it.source,
                "published": it.published,
                "url": it.url,
                "snippet": it.snippet[:300],
            }
        )

    sys_msg = (
        "You are a careful news editor. Output STRICT JSON only. "
        "No markdown, no prose outside JSON."
    )
    user_msg = {
        "section_id": section_id,
        "section_title_cn": section_title_cn,
        "task": "Pick the most important news items and write Chinese brief summaries.",
        "requirements": {
            "language": "zh",
            "items_per_section": items_per_section,
            "events_schema": {
                "id": "string",
                "title": "string (keep original title or rewrite in English)",
                "title_zh": "string (Chinese title)",
                "summary_zh": "string (<=140 Chinese chars, include what happened + why it matters)",
                "url": "string",
                "source_hint": "string",
                "date": "string (YYYY-MM-DD if possible)",
                "tags": ["string", "..."],
            },
            "return_schema": {"brief_zh": "string", "events": ["events_schema..."]},
        },
        "candidates": raw_payload,
    }

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
    ]

    try:
        resp = openai_chat_json(model, messages, timeout=90.0, tries=3)
        content = (
            resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        obj = _extract_json_object(content)
        if not obj:
            raise ValueError("OpenAI did not return JSON object")
        brief = (obj.get("brief_zh") or "").strip()
        events = obj.get("events") or []
        if not isinstance(events, list):
            events = []
        # normalize + cap
        norm_events: List[Dict[str, Any]] = []
        for e in events[:items_per_section]:
            if not isinstance(e, dict):
                continue
            url = (e.get("url") or "").strip()
            title = (e.get("title") or "").strip()
            title_zh = (e.get("title_zh") or "").strip()
            summary_zh = (e.get("summary_zh") or "").strip()
            source_hint = (e.get("source_hint") or "").strip()
            date = (e.get("date") or "").strip()
            tags = e.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            if not url and title:
                # try to map back by title
                for it in raw_items:
                    if it.title == title and it.url:
                        url = it.url
                        break
            if not url:
                continue
            eid = (e.get("id") or "").strip() or stable_id(section_id, url, title_zh or title)
            norm_events.append(
                {
                    "id": eid,
                    "title": title or title_zh or "",
                    "title_zh": title_zh or title or "",
                    "summary_zh": summary_zh,
                    "url": url,
                    "source_hint": source_hint,
                    "date": date,
                    "tags": [str(t) for t in tags[:6] if str(t).strip()],
                    "region": "",  # optional field used by UI filters
                }
            )
        return brief, norm_events
    except Exception as e:
        log(f"[digest] llm section pack failed ({section_id}): {e}")
        # Fallback: raw items -> events with minimal zh fields
        events = []
        for it in raw_items[:items_per_section]:
            events.append(
                {
                    "id": stable_id(section_id, it.url, it.title),
                    "title": it.title,
                    "title_zh": it.title,
                    "summary_zh": (it.snippet or "")[:140],
                    "url": it.url,
                    "source_hint": it.source,
                    "date": _date_only(it.published),
                    "tags": [],
                    "region": "",
                }
            )
        return "", events


def _date_only(published: str) -> str:
    d = parse_rfc2822(published)
    if d:
        return d.astimezone(_dt.timezone(_dt.timedelta(hours=8))).date().isoformat()
    # Atom timestamps
    try:
        d2 = _dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
        if d2.tzinfo is None:
            d2 = d2.replace(tzinfo=_dt.timezone.utc)
        return d2.astimezone(_dt.timezone(_dt.timedelta(hours=8))).date().isoformat()
    except Exception:
        return ""


# ----------------------------
# KPI fetch (best-effort)
# ----------------------------

def fetch_kpis(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    kpis_cfg = cfg.get("kpis") or []
    out: List[Dict[str, Any]] = []
    for k in kpis_cfg:
        if not isinstance(k, dict):
            continue
        kid = str(k.get("id") or k.get("name") or "")
        title = str(k.get("title") or k.get("name") or kid)
        unit = str(k.get("unit") or "")
        source = str(k.get("source") or "")
        out.append(
            {
                "id": kid,
                "title": title,
                "unit": unit,
                "value": None,
                "delta": None,
                "direction": None,
                "delta_pct": None,
                "source": source,
                "spark": "",
            }
        )
    return out


# ----------------------------
# Build digest
# ----------------------------

def normalize_sections(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections = cfg.get("sections") or []
    if isinstance(sections, dict):
        # legacy: {"macro": {...}, ...}
        out = []
        for sid, sc in sections.items():
            if isinstance(sc, dict):
                d = dict(sc)
                d.setdefault("id", sid)
                out.append(d)
        return out
    if isinstance(sections, list):
        return [s for s in sections if isinstance(s, dict)]
    return []


def build_raw_items_for_section(sec: Dict[str, Any], cfg: Dict[str, Any], *, limit_raw: int) -> List[RawItem]:
    sid = str(sec.get("id") or "")
    # Optional RSS feeds
    feeds = sec.get("feeds") or []
    # Google News queries
    gnews_queries = sec.get("gnews_queries") or sec.get("queries") or []

    locale = cfg.get("gnews_locale") or {}
    hl = str(locale.get("hl") or "en-US")
    gl = str(locale.get("gl") or "US")
    ceid = str(locale.get("ceid") or "US:en")

    raw: List[RawItem] = []
    seen = set()

    # RSS feeds
    for f in feeds:
        if isinstance(f, str):
            f_url = f
            f_src = ""
        elif isinstance(f, dict):
            f_url = str(f.get("url") or "")
            f_src = str(f.get("source") or "")
        else:
            continue
        if not f_url:
            continue

        def _do():
            xml = http_get(f_url, timeout=20.0)
            entries = parse_rss(xml, fallback_source=f_src)
            return entries

        try:
            entries = retry(_do, tries=2, base_sleep=1.0, label=f"feed {f_src or sid}")
        except Exception as e:
            log(f"[digest] feed {sid} failed: {e}")
            continue

        for e in entries:
            title = (e.get("title") or "").strip()
            url = (e.get("link") or "").strip()
            source = (e.get("source") or f_src or "").strip()
            published = (e.get("published") or "").strip()
            snippet = (e.get("summary") or "").strip()
            if not title or not url:
                continue
            key = url or title
            if key in seen:
                continue
            seen.add(key)
            raw.append(RawItem(title=title, url=url, source=source, published=published, snippet=snippet, section=sid))

    # Google News RSS
    for q in gnews_queries:
        if not isinstance(q, str):
            continue
        query = q.strip()
        if not query:
            continue
        url = google_news_rss_url(query, hl=hl, gl=gl, ceid=ceid)

        def _do():
            xml = http_get(url, timeout=20.0)
            return parse_rss(xml, fallback_source="Google News")

        try:
            entries = retry(_do, tries=2, base_sleep=1.0, label=f"gnews {sid}")
        except Exception as e:
            log(f"[digest] gnews {sid} failed: {e}")
            continue

        log(f"[digest] gnews {sid} q='{query[:26]}...' entries={len(entries)}")
        for e in entries:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            published = (e.get("published") or "").strip()
            source = (e.get("source") or "Google News").strip()
            snippet = (e.get("summary") or "").strip()
            if not title or not link:
                continue
            key = link
            if key in seen:
                continue
            seen.add(key)
            raw.append(RawItem(title=title, url=link, source=source, published=published, snippet=snippet, section=sid))

    # Sort by published desc (best-effort)
    def _key(it: RawItem):
        d = parse_rfc2822(it.published)
        if not d:
            # Atom ISO
            try:
                d = _dt.datetime.fromisoformat(it.published.replace("Z", "+00:00"))
                if d.tzinfo is None:
                    d = d.replace(tzinfo=_dt.timezone.utc)
            except Exception:
                d = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
        return d

    raw.sort(key=_key, reverse=True)
    return raw[:limit_raw]


def build_digest(cfg: Dict[str, Any], *, date: str, model: str, limit_raw: int, items_per_section: int) -> Dict[str, Any]:
    log("[digest] boot")
    log(f"[digest] date={date} model={model} limit_raw={limit_raw} items={items_per_section}")

    # KPI (currently best-effort stub; template accepts empty/null values)
    kpis_cfg = cfg.get("kpis") or []
    log(f"[digest] kpi start: {len(kpis_cfg) if isinstance(kpis_cfg, list) else 0}")
    kpis = fetch_kpis(cfg)
    log(f"[digest] kpi done: {len(kpis)}")

    sections_cfg = normalize_sections(cfg)
    log(f"[digest] sections start: {len(sections_cfg)}")

    out_sections: List[Dict[str, Any]] = []
    for sec in sections_cfg:
        sid = str(sec.get("id") or "")
        title_cn = str(sec.get("title_cn") or sec.get("title") or sid)
        log(f"[digest] section start: {sid} ({title_cn})")
        raw_items = build_raw_items_for_section(sec, cfg, limit_raw=limit_raw)
        if not raw_items:
            log(f"[digest] section {sid}: raw=0 (no candidates)")
            out_sections.append({"id": sid, "title_cn": title_cn, "brief_zh": "", "events": []})
            log(f"[digest] section done: {sid} events=0")
            continue

        brief_zh, events = llm_section_pack(model, sid, title_cn, raw_items, items_per_section=items_per_section)
        out_sections.append({"id": sid, "title_cn": title_cn, "brief_zh": brief_zh, "events": events})
        log(f"[digest] section done: {sid} events={len(events)}")

    log(f"[digest] sections done: {len(out_sections)}")

    digest: Dict[str, Any] = {
        "date": date,
        "generated_at_bjt": bjt_now().strftime("%Y-%m-%d %H:%M:%S"),
        "kpis": kpis,
        "alerts": [],
        "sections": out_sections,
    }
    return digest


# ----------------------------
# Render
# ----------------------------

def render_html(template_path: str, digest: Dict[str, Any]) -> str:
    tpl = open(template_path, "r", encoding="utf-8").read()
    # Primary mode for daily_digest_template_v15_apple.html
    if "__DIGEST_JSON__" in tpl:
        js_obj = json.dumps(digest, ensure_ascii=False)
        return tpl.replace("__DIGEST_JSON__", js_obj)

    # Fallback: if template is Jinja2-based (legacy)
    try:
        import jinja2
        from markupsafe import Markup
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=True)
        t = env.from_string(tpl)
        ctx = dict(digest)
        ctx["DIGEST_JSON"] = Markup(json.dumps(digest, ensure_ascii=False))
        ctx["DATE"] = digest.get("date")
        ctx["GENERATED_AT_BJT"] = digest.get("generated_at_bjt")
        # convenience aliases used by older templates
        ctx["TITLE"] = "Ben的每日资讯简报"
        return t.render(**ctx)
    except Exception:
        # As a last resort, return raw template.
        return tpl


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items_per_section", type=int, default=15)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    cfg = json.loads(open(args.config, "r", encoding="utf-8").read())
    digest = build_digest(cfg, date=args.date, model=args.model, limit_raw=args.limit_raw, items_per_section=args.items_per_section)
    log("[digest] render html")
    html = render_html(args.template, digest)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"[digest] wrote: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)