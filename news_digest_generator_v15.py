#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ben Daily Digest Generator (v15)
- Fetches sources server-side (Google News RSS + optional RSS feeds + FRED CSV)
- Summarizes in Chinese via OpenAI (single-call per section + one call for Top)
- Outputs a single static index.html by embedding JSON into the template.
No client-side fetching of news/data.

CLI:
  python news_digest_generator_v15.py --config digest_config_v15.json --template daily_digest_template_v15_apple.html --out index.html
Optional:
  --date YYYY-MM-DD
  --model gpt-4o-mini
  --limit_raw 25
  --cluster_threshold 0.82   (kept for backward-compat; used as "dedupe aggressiveness" only)
  --items_per_section 15
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import faulthandler
import json
import math
import os
import random
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import requests
import feedparser

# -----------------------------
# Robustness / diagnostics
# -----------------------------
def _enable_sigusr1_stackdump() -> None:
    """Allow 'kill -USR1 <pid>' to dump stack without terminating the process."""
    try:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        # Non-fatal; continue.
        pass

_enable_sigusr1_stackdump()


def log(msg: str) -> None:
    print(msg, flush=True)


def now_bjt() -> dt.datetime:
    # Beijing Time = UTC+8 (fixed offset)
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))


# -----------------------------
# HTTP helpers
# -----------------------------
UA = "Mozilla/5.0 (BenDailyDigest/1.0; +https://example.invalid) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

def http_get(url: str, *, timeout: Tuple[int, int] = (10, 25)) -> requests.Response:
    headers = {"User-Agent": UA, "Accept": "*/*"}
    return requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

def http_post(url: str, *, json_body: Dict[str, Any], timeout: Tuple[int, int] = (10, 60)) -> requests.Response:
    headers = {"User-Agent": UA, "Content-Type": "application/json", "Accept": "application/json"}
    return requests.post(url, headers=headers, json=json_body, timeout=timeout)

def retry(fn, *, tries=3, base_sleep=1.0, jitter=0.3, label="op"):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep = base_sleep * (1.6 ** i) + random.random() * jitter
            log(f"[digest] retry {label}: {type(e).__name__}: {e} (sleep {sleep:.1f}s)")
            time.sleep(sleep)
    raise last  # type: ignore


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class RawItem:
    title: str
    url: str
    source: str
    published: str  # ISO date string
    snippet: str = ""


@dataclass
class EventCard:
    id: str
    title: str
    title_zh: str
    summary_zh: str
    url: str
    source_hint: str
    date: str
    region: str  # "cn" or "global"


@dataclass
class SectionOut:
    id: str
    name: str
    tags: List[str]
    brief_zh: str
    brief_cn: str
    brief_global: str
    events: List[EventCard]


@dataclass
class KPI:
    id: str
    name: str
    unit: str
    value: Optional[float]
    delta: Optional[float]
    series: List[float]  # small trend series for sparkline
    source: str


# -----------------------------
# Source fetchers
# -----------------------------
def google_news_rss_url(query: str, *, hl="en-US", gl="US", ceid="US:en") -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

def normalize_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = re.sub(r"[\u2010-\u2015\-–—]", "-", t)
    # remove bracketed suffix
    t = re.sub(r"\s*[\(\[].{0,40}[\)\]]\s*$", "", t)
    return t

def domain(u: str) -> str:
    try:
        return urlparse(u).netloc.replace("www.", "")
    except Exception:
        return ""

def parse_rss_items(feed_url: str, source_name: str, max_items: int = 50) -> List[RawItem]:
    # feedparser can fetch itself, but we use requests for better timeout/retries.
    def _do():
        r = http_get(feed_url, timeout=(10, 25))
        r.raise_for_status()
        return r.text

    try:
        text = retry(_do, tries=2, base_sleep=0.8, label=f"feed {source_name}")
    except Exception as e:
        raise e

    parsed = feedparser.parse(text)
    items: List[RawItem] = []
    for ent in (parsed.entries or [])[:max_items]:
        title = getattr(ent, "title", "") or ""
        link = getattr(ent, "link", "") or ""
        if not title or not link:
            continue
        # prefer published date
        pub = getattr(ent, "published", "") or getattr(ent, "updated", "") or ""
        # normalize to YYYY-MM-DD if possible
        iso = ""
        if pub:
            try:
                # feedparser provides parsed struct_time
                st = getattr(ent, "published_parsed", None) or getattr(ent, "updated_parsed", None)
                if st:
                    iso = dt.datetime(*st[:6]).date().isoformat()
            except Exception:
                iso = ""
        if not iso:
            iso = now_bjt().date().isoformat()
        snippet = ""
        if hasattr(ent, "summary"):
            snippet = re.sub(r"<[^>]+>", "", getattr(ent, "summary", "") or "")
            snippet = re.sub(r"\s+", " ", snippet).strip()
        items.append(RawItem(title=title.strip(), url=link.strip(), source=source_name, published=iso, snippet=snippet))
    return items

def fetch_google_news_items(section_id: str, query: str, max_items: int = 60) -> List[RawItem]:
    url = google_news_rss_url(query)
    items = parse_rss_items(url, source_name=f"GNews:{section_id}", max_items=max_items)
    return items

def dedupe_raw(items: List[RawItem], max_keep: int) -> List[RawItem]:
    seen_title = set()
    seen_url = set()
    out: List[RawItem] = []
    for it in items:
        nt = normalize_title(it.title)
        if nt in seen_title:
            continue
        if it.url in seen_url:
            continue
        seen_title.add(nt)
        seen_url.add(it.url)
        out.append(it)
        if len(out) >= max_keep:
            break
    return out

# -----------------------------
# FRED KPIs
# -----------------------------
def fetch_fred_series_csv(series_id: str) -> List[Tuple[str, Optional[float]]]:
    # Public CSV endpoint (may occasionally 5xx)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote_plus(series_id)}"
    def _do():
        r = http_get(url, timeout=(10, 25))
        r.raise_for_status()
        return r.text

    text = retry(_do, tries=3, base_sleep=1.0, label=f"fred {series_id}")
    rows: List[Tuple[str, Optional[float]]] = []
    reader = csv.DictReader(text.splitlines())
    # columns: DATE, <SERIES_ID>
    col = series_id
    for row in reader:
        d = row.get("DATE")
        v = row.get(col)
        if not d:
            continue
        if v is None or v == "." or v == "":
            rows.append((d, None))
        else:
            try:
                rows.append((d, float(v)))
            except Exception:
                rows.append((d, None))
    return rows

def build_kpi(kcfg: Dict[str, Any], spark_points: int = 24) -> KPI:
    sid = kcfg.get("id") or kcfg.get("series") or ""
    name = kcfg.get("name", sid)
    unit = kcfg.get("unit", "")
    source = kcfg.get("source", "FRED")
    series_vals: List[float] = []
    value = None
    delta = None

    if source.upper() == "FRED" and sid:
        data = fetch_fred_series_csv(sid)
        # take last non-null points
        vals = [(d, v) for (d, v) in data if v is not None]
        if vals:
            value = vals[-1][1]
            if len(vals) >= 2:
                delta = vals[-1][1] - vals[-2][1]
            # build spark series
            tail = [v for (_, v) in vals[-spark_points:]]
            if tail:
                series_vals = tail
    else:
        # Placeholder for future sources
        value = None
        delta = None
        series_vals = []

    return KPI(id=sid, name=name, unit=unit, value=value, delta=delta, series=series_vals, source=source)

# -----------------------------
# OpenAI calls
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

def openai_chat_json(model: str, messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 900) -> Dict[str, Any]:
    """
    Use /v1/chat/completions with a strictly-JSON response expectation (no response_format to avoid model incompat).
    Returns parsed JSON dict.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    def _do():
        r = requests.post(url, headers=headers, json=body, timeout=(10, 80))
        if r.status_code >= 400:
            # include a short error for diagnosis (safe)
            raise requests.HTTPError(f"{r.status_code} {r.text[:500]}")
        return r.json()

    resp = retry(_do, tries=2, base_sleep=1.0, label="openai")
    text = resp["choices"][0]["message"]["content"]
    # Extract JSON object from the response
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("OpenAI did not return JSON object")
    return json.loads(m.group(0))

def safe_trunc(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def llm_section_pack(model: str, section_id: str, section_name: str, raw_items: List[RawItem], items_per_section: int) -> SectionOut:
    """
    One-call per section: select 10-20 event cards (结论优先), produce 300-500字简报 + tags.
    Also classify each card region as cn/global.
    """
    # Provide only essential fields to keep prompt small
    candidates = []
    for i, it in enumerate(raw_items[: max(10, min(60, len(raw_items)))]):
        candidates.append({
            "i": i + 1,
            "title": safe_trunc(it.title, 160),
            "source": it.source or domain(it.url),
            "date": it.published,
            "url": it.url,
            "snippet": safe_trunc(it.snippet or "", 220),
        })

    system = (
        "你是一名严谨的全球资讯编辑与研究员。你将从候选列表中筛选并聚合同一事件，"
        "输出“结论优先”的事件卡：先给结论，再给为什么重要。所有输出必须是中文。"
        "不要写空泛口号。每条事件卡的摘要要有可执行信息（谁做了什么、影响什么、下一步看什么）。"
    )

    user = {
        "section": {"id": section_id, "name": section_name},
        "requirements": {
            "n_events": items_per_section,
            "brief_len": "300-500字",
            "event_summary_len": "80-140字",
            "region_rule": "涉及中国政策/机构/企业/资产/供应链即标记cn，否则global",
        },
        "candidates": candidates,
        "output_schema": {
            "tags": ["string"],
            "brief_zh": "string (300-500字, 总览)",
            "brief_cn": "string (300-500字, 仅中国相关视角; 若不足请写'今日该板块暂无明确中国相关新增'并给出原因)",
            "brief_global": "string (300-500字, 不含中国的全球视角)",
            "events": [
                {
                    "title_zh": "string",
                    "summary_zh": "string",
                    "url": "string",
                    "source_hint": "string",
                    "date": "YYYY-MM-DD",
                    "region": "cn|global",
                }
            ],
        }
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        {"role": "user", "content": "请严格按 output_schema 输出一个 JSON 对象，不要包含任何多余文字。"},
    ]

    data = openai_chat_json(model, messages, temperature=0.25, max_tokens=1400)

    tags = [t for t in (data.get("tags") or []) if isinstance(t, str)][:8]
    brief_zh = (data.get("brief_zh") or "").strip()
    brief_cn = (data.get("brief_cn") or "").strip()
    brief_global = (data.get("brief_global") or "").strip()

    evs = []
    for idx, ev in enumerate((data.get("events") or [])[:items_per_section]):
        if not isinstance(ev, dict):
            continue
        url = str(ev.get("url") or "").strip()
        title_zh = str(ev.get("title_zh") or "").strip()
        if not url or not title_zh:
            continue
        evs.append(EventCard(
            id=f"{section_id}-{idx+1}-{abs(hash(url))%10_000_000}",
            title=str(ev.get("title") or title_zh),
            title_zh=title_zh,
            summary_zh=str(ev.get("summary_zh") or "").strip(),
            url=url,
            source_hint=str(ev.get("source_hint") or domain(url) or "").strip(),
            date=str(ev.get("date") or now_bjt().date().isoformat()),
            region=("cn" if str(ev.get("region") or "global").lower().startswith("c") else "global"),
        ))

    # Fallbacks to avoid empty sections / empty briefs
    if not brief_zh:
        brief_zh = "（今日该板块暂无可用内容：来源抓取或筛选不足，建议稍后重试或在设置中扩大候选来源。）"
    if not brief_cn:
        brief_cn = "今日该板块暂无明确中国相关新增。"
    if not brief_global:
        brief_global = brief_zh

    return SectionOut(
        id=section_id,
        name=section_name,
        tags=tags,
        brief_zh=brief_zh,
        brief_cn=brief_cn,
        brief_global=brief_global,
        events=evs,
    )


def llm_top_brief(model: str, sections: List[SectionOut], kpis: List[KPI]) -> str:
    """
    One-call for '今日要点' 300-500字，结构化且更实质。
    Keep prompt small: feed only top titles and KPI deltas.
    """
    # KPI summary lines
    kpi_lines = []
    for k in kpis[:9]:
        val = "NA" if k.value is None else f"{k.value:.3f}"
        if k.delta is None:
            dlt = "NA"
        else:
            dlt = f"{k.delta:+.3f}"
        kpi_lines.append(f"{k.name}: {val}{k.unit} (Δ {dlt})")

    sec_lines = []
    for s in sections:
        titles = [e.title_zh or e.title for e in s.events[:4]]
        sec_lines.append({"section": s.name, "titles": titles})

    system = (
        "你是一名全球市场与产业情报编辑。输出一段 300-500 字的中文“今日要点”，要求：\n"
        "1) 结论优先：先一句话给出今日主线；\n"
        "2) 三段结构：①主线与驱动 ②关键变量（利率/汇率/金属/风险偏好）③风险与机会（未来24-72小时关注点）；\n"
        "3) 避免空泛措辞，必须引用候选标题中的具体事实线索；\n"
        "4) 不要列清单式新闻串，保持可读性。"
    )
    user = {
        "date": now_bjt().date().isoformat(),
        "kpis": kpi_lines,
        "section_samples": sec_lines,
        "limit": "300-500字"
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        {"role": "user", "content": "只输出最终中文段落，不要加标题，不要加多余解释。"},
    ]
    try:
        data = openai_chat_json(model, messages, temperature=0.25, max_tokens=650)
        # The helper returns JSON; but here we asked plain text.
        # If model returned JSON anyway, extract string.
        if isinstance(data, dict):
            # try common keys
            for k in ["text", "brief", "output", "content"]:
                if isinstance(data.get(k), str) and data.get(k).strip():
                    return data[k].strip()
        return json.dumps(data, ensure_ascii=False)[:480]
    except Exception as e:
        log(f"[digest] llm daily brief failed: {e}")
        # Fallback
        return "今日主线：市场继续围绕利率路径、美元强弱与关键大宗波动定价，叠加地缘与制裁不确定性。关注点集中在美债收益率与美元对主要货币的边际变化、LME关键金属价格与库存信号，以及AI算力/数据中心电力与冷却约束的新增进展。接下来24-72小时重点盯紧：政策/制裁新增细节是否落地、风险资产情绪是否随宏观数据转向、以及供应链/关键矿产与碳合规（CBAM）规则更新对成本与交付的传导。"


# -----------------------------
# Build digest
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def build_raw_items_for_section(sec: Dict[str, Any], limit_raw: int) -> List[RawItem]:
    sid = sec["id"]
    raw: List[RawItem] = []

    # Feeds
    for f in sec.get("feeds", []) or []:
        try:
            if isinstance(f, str):
                src_name, url = (domain(f) or "RSS"), f
            elif isinstance(f, dict):
                src_name = str(f.get("source") or domain(str(f.get("url") or "")) or "RSS")
                url = str(f.get("url") or "")
            else:
                continue
            if not url.startswith("http"):
                continue
            items = parse_rss_items(url, source_name=src_name, max_items=50)
            raw.extend(items)
        except Exception as e:
            log(f"[digest] feed {sid} failed: {e}")

    # Google News queries
    for q in sec.get("queries", []) or []:
        if not isinstance(q, str) or not q.strip():
            continue
        try:
            items = fetch_google_news_items(sid, q.strip(), max_items=80)
            log(f"[digest] gnews {sid} q='{q[:24]}...' entries={len(items)}")
            raw.extend(items)
        except Exception as e:
            log(f"[digest] gnews {sid} failed: {e}")

    # Normalize / dedupe / keep recent-ish
    today = now_bjt().date()
    def _recency_score(d: str) -> int:
        try:
            dd = dt.date.fromisoformat(d)
            return abs((today - dd).days)
        except Exception:
            return 999

    raw.sort(key=lambda x: (_recency_score(x.published), len(x.title)))
    raw = dedupe_raw(raw, max_keep=max(limit_raw, 10))
    return raw

def build_digest(cfg: Dict[str, Any], *, date: str, model: str, limit_raw: int, items_per_section: int) -> Dict[str, Any]:
    log("[digest] boot")
    log(f"[digest] date={date} model={model} limit_raw={limit_raw} items={items_per_section}")

    # KPIs
    kpis_cfg = cfg.get("kpis", []) or []
    log(f"[digest] kpi start: {len(kpis_cfg)}")
    kpis: List[KPI] = []
    for kcfg in kpis_cfg:
        try:
            kpis.append(build_kpi(kcfg))
        except Exception as e:
            log(f"[digest] kpi failed {kcfg.get('id')}: {e}")
            kpis.append(KPI(id=str(kcfg.get("id","")), name=str(kcfg.get("name","")), unit=str(kcfg.get("unit","")), value=None, delta=None, series=[], source=str(kcfg.get("source",""))))
    log(f"[digest] kpi done: {len(kpis)}")

    # Sections
    sections_cfg = cfg.get("sections", []) or []
    log(f"[digest] sections start: {len(sections_cfg)}")
    sections: List[SectionOut] = []
    for sec in sections_cfg:
        sid = sec["id"]
        sname = sec["name"]
        log(f"[digest] section start: {sid} ({sname})")
        raw = build_raw_items_for_section(sec, limit_raw=limit_raw)
        if not raw:
            log(f"[digest] section {sid}: raw=0 (no candidates)")
            # still output placeholder section
            sections.append(SectionOut(id=sid, name=sname, tags=[], brief_zh="（今日该板块暂无可用内容：无候选来源。）", brief_cn="今日该板块暂无明确中国相关新增。", brief_global="（今日该板块暂无可用内容：无候选来源。）", events=[]))
            log(f"[digest] section done: {sid} events=0")
            continue
        try:
            sec_out = llm_section_pack(model, sid, sname, raw, items_per_section=items_per_section)
        except Exception as e:
            log(f"[digest] llm section failed {sid}: {e}")
            # fallback: take top URLs
            events = []
            for i, it in enumerate(raw[:items_per_section]):
                events.append(EventCard(
                    id=f"{sid}-raw-{i+1}-{abs(hash(it.url))%10_000_000}",
                    title=it.title,
                    title_zh=it.title,
                    summary_zh=safe_trunc(it.snippet or "（未生成摘要）", 120),
                    url=it.url,
                    source_hint=it.source or domain(it.url),
                    date=it.published,
                    region=("cn" if "china" in (it.title.lower()+it.snippet.lower()) else "global"),
                ))
            sec_out = SectionOut(
                id=sid, name=sname, tags=[],
                brief_zh="（LLM生成失败：已退化为候选列表直出。建议检查 OPENAI_API_KEY 与模型参数。）",
                brief_cn="今日该板块暂无明确中国相关新增。",
                brief_global="（LLM生成失败：已退化为候选列表直出。建议检查 OPENAI_API_KEY 与模型参数。）",
                events=events,
            )
        sections.append(sec_out)
        log(f"[digest] section done: {sid} events={len(sec_out.events)}")

    # Top brief section (as first section)
    top_text = llm_top_brief(model, sections, kpis)
    top_section = SectionOut(
        id="top",
        name="Top 今日要点",
        tags=["结论优先", "主线", "风险/机会"],
        brief_zh=top_text,
        brief_cn=top_text,       # keep consistent; region filter still works for events
        brief_global=top_text,
        events=[],               # no cards, only brief
    )
    sections = [top_section] + sections

    digest = {
        "date": date,
        "generated_at_bjt": now_bjt().strftime("%Y-%m-%d %H:%M (BJT)"),
        "kpis": [
            {
                "id": k.id, "name": k.name, "unit": k.unit, "value": k.value,
                "delta": k.delta, "series": k.series, "source": k.source
            } for k in kpis
        ],
        "sections": [
            {
                "id": s.id,
                "name": s.name,
                "tags": s.tags,
                "brief_zh": s.brief_zh,
                "brief_cn": s.brief_cn,
                "brief_global": s.brief_global,
                "events": [
                    {
                        "id": e.id,
                        "title": e.title,
                        "title_zh": e.title_zh,
                        "summary_zh": e.summary_zh,
                        "url": e.url,
                        "source_hint": e.source_hint,
                        "date": e.date,
                        "region": e.region,
                    } for e in s.events
                ],
            } for s in sections
        ],
        "meta": {
            "items_per_section": items_per_section,
            "limit_raw": limit_raw,
            "generator": "news_digest_generator_v15.py",
        }
    }
    log(f"[digest] sections done: {len(digest['sections'])}")
    return digest


# -----------------------------
# HTML output
# -----------------------------
def render_html(template_path: str, digest: Dict[str, Any]) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")
    payload = json.dumps(digest, ensure_ascii=False)
    if "__DIGEST_JSON__" not in tpl:
        # Backward compatibility: try window.DIGEST assignment marker
        return tpl + f"\n<script>window.DIGEST={payload};</script>\n"
    return tpl.replace("__DIGEST_JSON__", payload)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--date", default=None)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--cluster_threshold", type=float, default=0.82)  # retained but not used deeply
    ap.add_argument("--items_per_section", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    items_per_section = int(args.items_per_section or cfg.get("items_per_section") or 15)

    date = args.date or now_bjt().date().isoformat()

    digest = build_digest(
        cfg,
        date=date,
        model=args.model,
        limit_raw=args.limit_raw,
        items_per_section=items_per_section,
    )

    log("[digest] render html")
    html_out = render_html(args.template, digest)
    Path(args.out).write_text(html_out, encoding="utf-8")
    log(f"[digest] wrote: {args.out}")

if __name__ == "__main__":
    main()
