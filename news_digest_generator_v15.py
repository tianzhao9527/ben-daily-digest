#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py
- Fetches candidate items (primarily Google News RSS + optional RSS feeds)
- Builds "conclusion-first" event cards via a small number of LLM calls
- Renders a single static HTML page (no client-side fetching)

Designed to run reliably on GitHub Actions.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import json
import os
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import requests
import feedparser
from jinja2 import Environment, BaseLoader, StrictUndefined
import faulthandler

# ----------------------------
# Safety / debugging hooks
# ----------------------------
# Allow the workflow to send SIGUSR1 to dump stacks without killing the process.
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    pass

UA = os.getenv("DIGEST_UA", "Mozilla/5.0 (compatible; BenDailyDigest/1.0; +https://github.com/)")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_TIMEOUT = (10, 60)  # connect, read


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------
# Data structures
# ----------------------------
@dataclasses.dataclass
class RawItem:
    title: str
    url: str
    source: str = ""
    published: str = ""
    snippet: str = ""
    section: str = ""


@dataclasses.dataclass
class KPI:
    key: str
    name: str
    value: float
    unit: str = ""
    delta: Optional[float] = None
    spark: str = ""  # svg path or inline sparkline string
    hint: str = ""


# ----------------------------
# Helpers
# ----------------------------
def retry(fn, *, tries: int = 3, base: float = 0.8, jitter: float = 0.5, what: str = ""):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep = base * (1.6 ** i) + random.random() * jitter
            log(f"[digest] retry {what}: {type(e).__name__}: {e} (sleep {sleep:.1f}s)")
            time.sleep(sleep)
    raise last  # type: ignore


def http_get(url: str, *, timeout=DEFAULT_TIMEOUT, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    r = requests.get(url, timeout=timeout, headers=h)
    r.raise_for_status()
    return r


def http_post(url: str, *, json_body: Dict[str, Any], timeout=DEFAULT_TIMEOUT, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    h = {"User-Agent": UA, "Content-Type": "application/json"}
    if headers:
        h.update(headers)
    r = requests.post(url, timeout=timeout, headers=h, data=json.dumps(json_body))
    r.raise_for_status()
    return r


def today_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def norm_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[“”\"'’]", "", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_id(*parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


def is_cn_related(text: str, url: str = "") -> bool:
    t = (text or "").lower()
    # conservative rules: only mark "China-related" if strongly indicated
    keys = [
        "china", "chinese", "beijing", "shanghai", "shenzhen", "hong kong", "taiwan",
        "prc", "cny", "pbo c", "pboC", "xinjiang", "guangdong", "tianjin", "wuhan",
        "hainan", "macau", "taipei", "renminbi",
    ]
    if any(k in t for k in keys):
        return True
    try:
        host = urlparse(url).netloc.lower()
        if host.endswith(".cn"):
            return True
    except Exception:
        pass
    return False


# ----------------------------
# Sources
# ----------------------------
def google_news_rss_url(q: str) -> str:
    # Use US English to avoid region bias; still global coverage.
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"


def fetch_google_news_items(query: str, *, section_id: str, limit: int = 40) -> List[RawItem]:
    url = google_news_rss_url(query)
    def _do():
        r = http_get(url)
        fp = feedparser.parse(r.text)
        out: List[RawItem] = []
        for e in fp.entries[:limit]:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            # source is often like "Reuters" in e.source.title
            src = ""
            try:
                src = e.source.title  # type: ignore
            except Exception:
                pass
            published = getattr(e, "published", "") or ""
            summary = getattr(e, "summary", "") or ""
            out.append(RawItem(title=title, url=link, source=src, published=published, snippet=summary, section=section_id))
        return out
    return retry(_do, tries=3, what=f"gnews {section_id}")


def parse_rss_items(feed_url: str, *, section_id: str, limit: int = 40, source_name: str = "") -> List[RawItem]:
    def _do():
        r = http_get(feed_url)
        fp = feedparser.parse(r.text)
        out: List[RawItem] = []
        for e in fp.entries[:limit]:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            published = getattr(e, "published", "") or ""
            summary = getattr(e, "summary", "") or ""
            src = source_name
            out.append(RawItem(title=title, url=link, source=src, published=published, snippet=summary, section=section_id))
        return out
    return retry(_do, tries=2, what=f"feed {source_name or section_id}")


# ----------------------------
# KPIs (lightweight, no paywalls)
# ----------------------------
def fred_series_csv(series_id: str) -> str:
    # FRED provides a CSV endpoint for graph data.
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_fred_series(series_id: str, *, n: int = 30) -> List[Tuple[str, float]]:
    def _do():
        r = http_get(fred_series_csv(series_id), timeout=(10, 40))
        rows = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        # header: DATE,VALUE
        pts: List[Tuple[str, float]] = []
        for ln in rows[1:]:
            try:
                d, v = ln.split(",", 1)
                if v == ".":
                    continue
                pts.append((d, float(v)))
            except Exception:
                continue
        return pts[-n:]
    return retry(_do, tries=3, what=f"fred {series_id}")


def sparkline_svg(values: List[float], *, w: int = 140, h: int = 34, pad: int = 3) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        mx = mn + 1.0
    xs = []
    ys = []
    for i, v in enumerate(values):
        x = pad + (w - 2 * pad) * (i / max(1, len(values) - 1))
        y = pad + (h - 2 * pad) * (1 - (v - mn) / (mx - mn))
        xs.append(x)
        ys.append(y)
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"><path d="{d}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'



def group_kpis_for_template(kpis: list[KPI]) -> dict:
    """Transform KPI list into the dict schema used by the Apple-style template."""
    out = {"metals": [], "macro": []}
    for k in kpis:
        d = dataclasses.asdict(k)
        # Normalize/rename for template
        item = {
            "label": d.get("name") or d.get("key"),
            "value_str": d.get("value_str") or ("" if d.get("value") is None else str(d.get("value"))),
            "unit": d.get("unit") or "",
            "delta": d.get("delta"),
            "pct": d.get("delta_pct"),
            "source": d.get("source") or "",
            "spark_svg": d.get("spark_svg") or "",
        }
        key = (d.get("key") or "").lower()
        if key.startswith("p_"):  # proxy series
            out["metals"].append(item)
        elif "lme" in key or "metal" in key:
            out["metals"].append(item)
        else:
            out["macro"].append(item)
    return out

def build_kpis(kpi_cfg: List[Dict[str, Any]]) -> List[KPI]:
    out: List[KPI] = []
    for k in kpi_cfg:
        sid = k.get('fred') or k.get('series') or k.get('id')
        if not sid:
            continue
        pts = fetch_fred_series(sid, n=30)
        vals = [v for _, v in pts]
        if not vals:
            continue
        value = vals[-1]
        delta = (vals[-1] - vals[-2]) if len(vals) >= 2 else None
        out.append(
            KPI(
                key=(k.get('key') or sid),
                name=(k.get('name') or k.get('title') or sid),
                value=value,
                unit=k.get("unit", ""),
                delta=delta,
                spark=sparkline_svg(vals),
                hint=k.get("hint", ""),
            )
        )
    return out


# ----------------------------
# OpenAI JSON helpers
# ----------------------------
def openai_chat_json(messages: List[Dict[str, str]], *, model: str, max_output_tokens: int = 900) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "max_tokens": max_output_tokens,
    }
    def _do():
        r = http_post(url, json_body=body, timeout=(10, 120), headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
        data = r.json()
        txt = data["choices"][0]["message"]["content"]
        try:
            return json.loads(txt)
        except Exception as e:
            raise ValueError(f"OpenAI did not return JSON object: {e} :: {txt[:200]}")
    return retry(_do, tries=2, what="openai")


def ensure_brief(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure the template-required keys exist.
    return {
        "one_liner": obj.get("one_liner", "") or "",
        "facts": list(obj.get("facts", []) or []),
        "impacts": list(obj.get("impacts", []) or []),
        "actions": list(obj.get("actions", []) or []),
        "watch": list(obj.get("watch", []) or []),
        "summary": obj.get("summary", "") or "",
    }


def ensure_top(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "one_liner": obj.get("one_liner", "") or "",
        "facts": list(obj.get("facts", []) or []),
        "actions": list(obj.get("actions", []) or []),
        "watch": list(obj.get("watch", []) or []),
        "summary": obj.get("summary", "") or "",
    }


# ----------------------------
# LLM packing: "结论优先"
# ----------------------------
def llm_section_pack(section_title: str, items: List[RawItem], *, model: str, want_n: int) -> Dict[str, Any]:
    # Keep the payload bounded to avoid 400 errors.
    items = items[: max(want_n * 3, 30)]
    payload = []
    for it in items:
        payload.append({
            "title": it.title[:160],
            "url": it.url,
            "source": it.source[:60],
            "published": it.published[:60],
        })
    prompt = {
        "section": section_title,
        "target_cards": want_n,
        "items": payload,
    }
    sys_msg = (
        "你是一个严谨的财经/产业情报编辑。"
        "基于给定链接列表（可能有同一事件的多篇报道），生成“结论优先”的事件卡。"
        "要求：中文输出；每个事件卡包含：title_zh（<=26字）、conclusion（1句结论）、"
        "facts（2-4条要点）、impact（对业务/市场含义 1-2条）、actions（可执行动作 1-2条）、"
        "watch（风险/观察点 1-2条）、links（1-3条原文链接）。"
        "总共输出 target_cards 条事件卡；如聚类后不足，则用剩余单篇补足。"
        "并输出一个 section_brief：300-500字，结构化（结论/关键事实/影响/动作/观察），避免空话。"
        "只返回JSON。"
    )
    user_msg = json.dumps(prompt, ensure_ascii=False)
    j = openai_chat_json(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        model=model,
        max_output_tokens=1100,
    )
    cards = j.get("cards", []) or []
    section_brief = ensure_brief(j.get("section_brief", {}) or {})
    # Normalize cards; fill if short
    norm_cards: List[Dict[str, Any]] = []
    used_urls = set()
    for c in cards:
        links = c.get("links", []) or []
        links2 = []
        for lk in links:
            if isinstance(lk, str):
                links2.append({"title": "", "url": lk})
            elif isinstance(lk, dict) and lk.get("url"):
                links2.append({"title": lk.get("title","")[:80], "url": lk["url"]})
        if not links2:
            continue
        for lk in links2:
            used_urls.add(lk["url"])
        norm_cards.append({
            "id": stable_id(section_title, links2[0]["url"]),
            "title": c.get("title_zh", "") or c.get("title", "") or "",
            "conclusion": c.get("conclusion", "") or "",
            "facts": list(c.get("facts", []) or []),
            "impact": list(c.get("impact", []) or []),
            "actions": list(c.get("actions", []) or []),
            "watch": list(c.get("watch", []) or []),
            "links": links2[:3],
            "cn_related": any(is_cn_related((c.get("title_zh","") or "") + " " + (c.get("conclusion","") or ""), lk["url"]) for lk in links2),
        })
    # Fill missing cards with simple 1:1 mapping from unused items
    for it in items:
        if len(norm_cards) >= want_n:
            break
        if it.url in used_urls:
            continue
        norm_cards.append({
            "id": stable_id(section_title, it.url),
            "title": it.title,  # if not translated, keep English
            "conclusion": "",
            "facts": [],
            "impact": [],
            "actions": [],
            "watch": [],
            "links": [{"title": it.source, "url": it.url}],
            "cn_related": is_cn_related(it.title, it.url),
        })
        used_urls.add(it.url)
    return {"cards": norm_cards[:want_n], "brief": section_brief}


def llm_top_brief(sections: List[Dict[str, Any]], kpis: List[KPI], *, model: str) -> Dict[str, Any]:
    # Build compact input
    sec_in = []
    for s in sections:
        b = s.get("brief", {}) or {}
        sec_in.append({
            "name": s.get("name", ""),
            "one_liner": b.get("one_liner", ""),
            "facts": (b.get("facts", []) or [])[:2],
            "impact": (b.get("impacts", []) or [])[:2],
        })
    kpi_in = []
    for k in kpis[:10]:
        kpi_in.append({
            "name": k.name,
            "value": k.value,
            "unit": k.unit,
            "delta": k.delta,
        })
    prompt = {"sections": sec_in, "kpis": kpi_in}
    sys_msg = (
        "你是一个严谨的情报主编。请基于各栏目简报与KPI，输出今天的“今日要点”（300-500字），"
        "并给出结构化列表：facts(3-5条)、actions(2-4条)、watch(2-4条)与一句话one_liner。"
        "避免空泛形容词，优先写可验证的事实、明确的影响与可执行动作。只返回JSON。"
    )
    j = openai_chat_json(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        model=model,
        max_output_tokens=900,
    )
    return ensure_top(j)


# ----------------------------
# Candidate selection / dedupe
# ----------------------------
def dedupe_items(items: List[RawItem], *, hard_limit: int) -> List[RawItem]:
    seen = set()
    out = []
    for it in items:
        key = (norm_title(it.title), it.url.split("?")[0])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= hard_limit:
            break
    return out


# ----------------------------
# Digest build
# ----------------------------
def build_raw_items_for_section(section_id: str, section_cfg: Dict[str, Any], *, limit_raw: int) -> List[RawItem]:
    items: List[RawItem] = []
    # RSS feeds (optional)
    for f in section_cfg.get("feeds", []) or []:
        url = f.get("url") if isinstance(f, dict) else str(f)
        name = f.get("source", "") if isinstance(f, dict) else ""
        if not url:
            continue
        try:
            got = parse_rss_items(url, section_id=section_id, limit=limit_raw, source_name=name)
            items.extend(got)
        except Exception as e:
            log(f"[digest] feed {section_id} failed: {e}")
    # Google News queries
    for q in section_cfg.get("queries", []) or []:
        try:
            got = fetch_google_news_items(q, section_id=section_id, limit=limit_raw)
            log(f"[digest] gnews {section_id} q='{q[:24]}...' entries={len(got)}")
            items.extend(got)
        except Exception as e:
            log(f"[digest] gnews {section_id} failed: {e}")
    items = dedupe_items(items, hard_limit=max(limit_raw, 30))
    return items


def build_digest(cfg: Dict[str, Any], *, date: str, model: str, limit_raw: int, items_per_section: int) -> Dict[str, Any]:
    # KPI panel
    log(f"[digest] kpi start: {len(cfg.get('kpis', []) or [])}")
    kpis = build_kpis(cfg.get("kpis", []) or [])
    log(f"[digest] kpi done: {len(kpis)}")

    # Sections
    sections_out: List[Dict[str, Any]] = []
    sections_cfg_raw = cfg.get("sections", []) or []
    # 支持两种格式：
    # 1) {"macro": {...}, "sanctions": {...}}  (dict)
    # 2) [{"id": "macro", ...}, {"id": "sanctions", ...}]  (list) —— digest_config_v15.json 使用该格式
    if isinstance(sections_cfg_raw, dict):
        sections_iter = list(sections_cfg_raw.items())
    else:
        sections_iter = []
        for i, sc in enumerate(list(sections_cfg_raw)):
            if not isinstance(sc, dict):
                continue
            sid = sc.get("id") or sc.get("key") or f"sec{i}"
            sections_iter.append((sid, sc))
    log(f"[digest] sections start: {len(sections_iter)}")
    for sid, sc in sections_iter:
        name = sc.get("name", sid)
        log(f"[digest] section start: {sid} ({name})")
        raw = build_raw_items_for_section(sid, sc, limit_raw=limit_raw)
        if not raw:
            log(f"[digest] section {sid}: raw=0 (no candidates)")
            sections_out.append({
                "id": sid,
                "name": name,
                "events": [],
                "brief": ensure_brief({}),
            })
            continue
        # Select candidates
        raw = raw[: max(limit_raw, items_per_section * 3)]
        try:
            pack = llm_section_pack(name, raw, model=model, want_n=items_per_section)
            cards = pack["cards"]
            brief = ensure_brief(pack.get("brief", {}) or {})
        except Exception as e:
            log(f"[digest] llm section pack failed ({sid}): {e}")
            # Fallback: keep first N as simple cards
            cards = [{
                "id": stable_id(sid, it.url),
                "url": it.url,
                "title": it.title,
                "title_zh": "",
                "summary": it.summary or "",
                "summary_zh": "",
                "source_hint": it.source,
                "source": it.source,
                "date": it.published,
                "region": ("cn" if is_cn_related(it.title, it.url) else "global"),
                "tags": [],
                "conclusion": "",
                "facts": [],
                "impact": [],
                "actions": [],
                "watch": [],
                "links": [{"title": it.source, "url": it.url}],
                "cn_related": is_cn_related(it.title, it.url),
            } for it in raw[:items_per_section]]
            brief = ensure_brief({})
        # Normalize event schema for JS template consumption
        for ev in cards:
            if not ev.get("url"):
                links = ev.get("links") or []
                if links and isinstance(links, list) and isinstance(links[0], dict):
                    ev["url"] = links[0].get("url") or ""
            if not ev.get("source_hint"):
                links = ev.get("links") or []
                if links and isinstance(links, list) and isinstance(links[0], dict):
                    ev["source_hint"] = links[0].get("title") or ""
            ev.setdefault("source", ev.get("source_hint") or "")
            ev.setdefault("date", ev.get("date") or "")
            ev.setdefault("region", ("cn" if ev.get("cn_related") else "global"))
            ev.setdefault("tags", [])
            ev.setdefault("title_zh", ev.get("title_zh") or "")
            ev.setdefault("summary", ev.get("summary") or "")
            ev.setdefault("summary_zh", ev.get("summary_zh") or "")

        sections_out.append({
            "id": sid,
            "name": name,
            "events": cards,
            "brief": brief,
        })
        log(f"[digest] section done: {sid} events={len(cards)}")
    log(f"[digest] sections done: {len(sections_out)}")

    # Views: cn / global(ex-cn)
    def filt_sections(pred):
        out = []
        for s in sections_out:
            ev = [e for e in (s.get("events", []) or []) if pred(e)]
            out.append({**s, "events": ev})
        return out

    cn_sections = filt_sections(lambda e: bool(e.get("cn_related")))
    gl_sections = filt_sections(lambda e: not bool(e.get("cn_related")))

    # Top briefs per view
    try:
        top_all = llm_top_brief(sections_out, kpis, model=model)
    except Exception as e:
        log(f"[digest] llm daily brief failed: {e}")
        top_all = ensure_top({})

    # For cn/global, do lightweight: reuse all-top but adjust one-liner
    top_cn = dict(top_all)
    top_gl = dict(top_all)
    if top_cn.get("one_liner"):
        top_cn["one_liner"] = "（中国相关）" + top_cn["one_liner"]
    if top_gl.get("one_liner"):
        top_gl["one_liner"] = "（全球·不含中国）" + top_gl["one_liner"]

    views = {
        "global": {"key": "global", "label": "全球（不含中国）", "top": top_gl, "sections": gl_sections},
        "cn": {"key": "cn", "label": "中国相关", "top": top_cn, "sections": cn_sections},
    }

    return {
        # New schema (top-level) expected by daily_digest_template_v15_apple.html (JS renderer)
        "title": cfg.get("title") or "Ben的每日资讯简报",
        "date": date,
        "generated_at_bjt": dt.datetime.utcnow().replace(
            tzinfo=dt.timezone(dt.timedelta(hours=8))
        ).isoformat(timespec="seconds"),
        # KPI groups for right-side panel
        "kpis": group_kpis_for_template(kpis),
        # Sections for main content
        "sections": sections_out,
        # Optional alerts (threshold triggers etc.)
        "alerts": [],
        # Backward-compatible fields (older templates / debugging)
        "meta": {
            "title": cfg.get("title") or "Ben的每日资讯简报",
            "date": date,
            "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "generated_at_bjt": dt.datetime.utcnow().replace(
                tzinfo=dt.timezone(dt.timedelta(hours=8))
            ).isoformat(timespec="seconds"),
            "model": model,
            "items_per_section": items_per_section,
        },
        "kpis_flat": [dataclasses.asdict(k) for k in kpis],
        "views": views,
        "sections_all": sections_out,
        "top": top,
    }
def render_html(template_path: str, digest: Dict[str, Any]) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")
    env = Environment(loader=BaseLoader(), autoescape=True, undefined=StrictUndefined)
    t = env.from_string(tpl)

    # Provide template variables (match v15 template expectations)
    context = {
        # Headline meta (template expects these as UPPERCASE)
        "TITLE": (digest.get("meta", {}) or {}).get("title") or "Ben的每日资讯简报",
        "DATE": (digest.get("meta", {}) or {}).get("date") or "",
        "GENERATED_AT": (digest.get("meta", {}) or {}).get("generated_at_bjt")
            or (digest.get("meta", {}) or {}).get("generated_at_utc")
            or "",

        # Backward/forward compatibility
        "META": digest.get("meta", {}),
        "KPIS": digest.get("kpis", []),

        # v15 template compatibility: some revisions used VIEWS, some used views
        "VIEWS": digest.get("views", {}) or {},
        "views": digest.get("views", {}) or {},
            "DIGEST_JSON": json.dumps(digest, ensure_ascii=False),
}

    # Defensive: ensure every view has required keys
    for _, v in (context["views"] or {}).items():
        v["top"] = ensure_top(v.get("top", {}) or {})
        for s in v.get("sections", []) or []:
            s["brief"] = ensure_brief(s.get("brief", {}) or {})
    return t.render(**context)


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--date", default=today_utc())
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items_per_section", type=int, default=15)
    args = ap.parse_args()

    log("[digest] boot")
    log(f"[digest] date={args.date} model={args.model} limit_raw={args.limit_raw} items={args.items_per_section}")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    digest = build_digest(cfg, date=args.date, model=args.model, limit_raw=args.limit_raw, items_per_section=args.items_per_section)

    log("[digest] render html")
    out_html = render_html(args.template, digest)
    Path(args.out).write_text(out_html, encoding="utf-8")
    log(f"[digest] wrote: {args.out}")


if __name__ == "__main__":
    main()
