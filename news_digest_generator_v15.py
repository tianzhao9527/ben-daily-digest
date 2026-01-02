#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ben Daily Digest Generator (v15, stability-hardened)

Key fixes for GitHub Actions stability:
- OpenAI calls: hard timeout + retries + graceful degradation (never hang forever)
- Reduce LLM calls: rule-based region classification + translation cache + LLM budget
- Strong progress logging (flush=True) for Actions
- All network calls use timeouts (requests.get / requests.post)
- Produces a single static HTML via Jinja2 template (no client-side fetching)

Usage:
  python -u news_digest_generator_v15.py \
    --config digest_config_v15.json \
    --template daily_digest_template_v15_apple.html \
    --out index.html \
    --limit_raw 25

Env:
  OPENAI_API_KEY: required for best results; if missing, falls back to extractive templates.
  OPENAI_MODEL: optional (default: gpt-4o-mini)  (change to your preferred model)
  OPENAI_BASE_URL: optional (default: https://api.openai.com/v1)
  DIGEST_LLM_BUDGET: optional integer (default: 40)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import requests
from jinja2 import Environment, FileSystemLoader, Undefined

# -----------------------------
# Global safety & logging
# -----------------------------

print("[digest] boot", flush=True)

# Enable SIGUSR1 stack dump (useful with watchdog)
try:
    import faulthandler
    import signal

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    pass

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Hard timeouts
HTTP_TIMEOUT_GET = (8, 25)   # connect, read
HTTP_TIMEOUT_POST = (10, 60) # connect, read

# Requests session
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# LLM settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

DEFAULT_LLM_BUDGET = int(os.environ.get("DIGEST_LLM_BUDGET", "40"))
_llm_calls_used = 0

# Translation cache (in-memory; optional disk persistence)
_ZH_CACHE: Dict[str, str] = {}
_CACHE_PATH = ".digest_cache_zh.json"

def _load_cache() -> None:
    global _ZH_CACHE
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _ZH_CACHE = json.load(f)
            print(f"[digest] loaded zh cache: {len(_ZH_CACHE)}", flush=True)
        except Exception as e:
            print(f"[digest] cache load failed: {e}", flush=True)

def _save_cache() -> None:
    # Best-effort; safe to ignore failures in Actions
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_ZH_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# -----------------------------
# Data models
# -----------------------------

@dataclasses.dataclass
class RawItem:
    title: str
    link: str
    source: str
    published: Optional[dt.datetime] = None
    summary: str = ""
    section_id: str = ""
    region: str = "all"  # all | cn | global
    tags: List[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class EventCard:
    event_id: str
    title_zh: str
    conclusion_zh: str
    evidence_zh: List[str]
    impact_zh: str
    next_step_zh: str
    sources: List[Tuple[str, str]]  # (source_name, url)
    published: Optional[dt.datetime]
    region: str  # cn | global | all
    score: float

@dataclasses.dataclass
class SectionOutput:
    section_id: str
    name: str
    brief_zh: str  # 300-500 chars
    events: List[EventCard]

@dataclasses.dataclass
class KPI:
    kpi_id: str
    name: str
    unit: str
    value: Optional[float]
    delta: Optional[float]  # latest - prev
    spark: List[float]
    status: str  # up|down|flat|na

@dataclasses.dataclass
class DigestViews:
    all: Dict[str, Any]
    cn: Dict[str, Any]
    global_: Dict[str, Any]


# -----------------------------
# Utility functions
# -----------------------------

def now_bj() -> dt.datetime:
    # Beijing = UTC+8
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))

def to_date_str(d: dt.date) -> str:
    return d.isoformat()

def safe_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def shorten_cn(s: str, min_chars: int = 300, max_chars: int = 500) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) < min_chars:
        return s
    if len(s) > max_chars:
        return s[:max_chars].rstrip() + "…"
    return s

def strip_html(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", txt).strip()

def parse_dt_any(entry: Dict[str, Any]) -> Optional[dt.datetime]:
    # feedparser supplies published_parsed / updated_parsed in time.struct_time
    for k in ("published_parsed", "updated_parsed"):
        v = entry.get(k)
        if v:
            try:
                return dt.datetime(*v[:6], tzinfo=dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))
            except Exception:
                pass
    return None

def token_set(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    toks = [t for t in s.split() if t and len(t) > 1]
    return set(toks)

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def similar_title(a: str, b: str) -> float:
    # cheap similarity: token jaccard + prefix bonus
    sa, sb = token_set(a), token_set(b)
    jac = jaccard(sa, sb)
    if a[:18].lower() == b[:18].lower():
        jac = max(jac, 0.85)
    return jac


# -----------------------------
# Region classification (rule-first)
# -----------------------------

CN_KEYWORDS = [
    "china", "chinese", "beijing", "shanghai", "shenzhen", "hong kong", "taiwan",
    "prc", "cny", "yuan", "renminbi", "pla",
    "中国", "大陆", "北京", "上海", "深圳", "香港", "台湾", "人民币", "央行", "商务部", "发改委"
]

CN_ENTITIES = [
    "pboC", "people's bank of china", "mofcom", "ndrc", "csrc", "state council",
    "huawei", "byd", "smic", "tencent", "alibaba", "temu", "shein",
    "中芯", "华为", "比亚迪", "腾讯", "阿里", "字节", "宁德时代", "中兴"
]

def classify_region_rule(text: str) -> str:
    t = (text or "").lower()
    # quick keyword hit
    for kw in CN_KEYWORDS:
        if kw.lower() in t:
            return "cn"
    for ent in CN_ENTITIES:
        if ent.lower() in t:
            return "cn"
    return "global"


# -----------------------------
# Fetchers
# -----------------------------

def http_get(url: str, timeout: Tuple[int, int] = HTTP_TIMEOUT_GET) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[digest] GET fail: {url} | {type(e).__name__}: {e}", flush=True)
        return None

def fetch_rss(url: str, source_name: str, limit: int) -> List[RawItem]:
    items: List[RawItem] = []
    try:
        # feedparser can parse from URL directly but may not respect our session headers reliably
        xml = http_get(url)
        if not xml:
            return items
        feed = feedparser.parse(xml)
        for entry in feed.entries[:limit]:
            title = safe_text(entry.get("title"))
            link = safe_text(entry.get("link"))
            summary = safe_text(entry.get("summary")) or safe_text(entry.get("description"))
            published = parse_dt_any(entry) or None
            if not title or not link:
                continue
            items.append(RawItem(
                title=strip_html(title),
                link=link,
                source=source_name,
                published=published,
                summary=strip_html(summary),
            ))
    except Exception as e:
        print(f"[digest] RSS parse fail: {url} | {type(e).__name__}: {e}", flush=True)
    return items

def fetch_fred_csv(series_id: str, points: int = 30) -> List[Tuple[dt.date, float]]:
    # No API key needed: fredgraph.csv
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    txt = http_get(url)
    if not txt:
        return []
    rows = txt.strip().splitlines()
    out: List[Tuple[dt.date, float]] = []
    for line in rows[1:]:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        ds, vs = parts[0].strip(), parts[1].strip()
        if vs == "." or vs == "":
            continue
        try:
            d = dt.date.fromisoformat(ds)
            v = float(vs)
            out.append((d, v))
        except Exception:
            continue
    return out[-points:]


# -----------------------------
# OpenAI (requests.post) with timeout + retries + degrade
# -----------------------------

def _llm_budget_ok(budget: int) -> bool:
    global _llm_calls_used
    return _llm_calls_used < budget

def openai_chat(messages: List[Dict[str, str]], model: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
    """
    Returns assistant content string. Never blocks indefinitely.
    On failure returns empty string (caller should degrade).
    """
    global _llm_calls_used

    if not OPENAI_API_KEY:
        return ""

    if not _llm_budget_ok(DEFAULT_LLM_BUDGET):
        return ""

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            _llm_calls_used += 1
            r = SESSION.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_POST)
            r.raise_for_status()
            data = r.json()
            return safe_text(data["choices"][0]["message"]["content"])
        except Exception as e:
            last_err = e
            # exponential backoff
            time.sleep(2 ** (attempt - 1))

    print(f"[digest] openai_chat failed: {type(last_err).__name__}: {last_err}", flush=True)
    return ""

def llm_json(system: str, user: str, schema_hint: str = "", max_tokens: int = 700) -> Dict[str, Any]:
    """
    Ask LLM to return JSON. If fails, return {}.
    """
    content = openai_chat(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + ("\n\nJSON schema hint:\n" + schema_hint if schema_hint else "")},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    if not content:
        return {}

    # Try to extract JSON block
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def llm_text(system: str, user: str, max_tokens: int = 700) -> str:
    return openai_chat(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.25,
    )


# -----------------------------
# Translation (cached)
# -----------------------------

def to_zh(text: str) -> str:
    t = safe_text(text)
    if not t:
        return ""
    if re.search(r"[\u4e00-\u9fff]", t):
        return t

    key = sha1(t)
    if key in _ZH_CACHE:
        return _ZH_CACHE[key]

    # If no API key / budget: fallback (no translation)
    if not OPENAI_API_KEY or not _llm_budget_ok(DEFAULT_LLM_BUDGET):
        _ZH_CACHE[key] = t
        return t

    sys_prompt = "你是专业新闻编辑与同传翻译。把英文标题/短句翻译成简洁准确的中文，不要加解释。"
    zh = llm_text(sys_prompt, t, max_tokens=120).strip()
    if not zh:
        zh = t
    _ZH_CACHE[key] = zh
    return zh


# -----------------------------
# Clustering into events
# -----------------------------

def cluster_items(items: List[RawItem], threshold: float) -> List[List[RawItem]]:
    clusters: List[List[RawItem]] = []
    for it in items:
        placed = False
        for c in clusters:
            if similar_title(it.title, c[0].title) >= threshold:
                c.append(it)
                placed = True
                break
        if not placed:
            clusters.append([it])
    # sort sources inside cluster by published time desc
    for c in clusters:
        c.sort(key=lambda x: x.published or dt.datetime(1970, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=8))), reverse=True)
    return clusters


# -----------------------------
# Event card generation (LLM-limited, conclusion-first)
# -----------------------------

def event_id_from_cluster(cluster: List[RawItem], section_id: str) -> str:
    base = section_id + "|" + "|".join(sorted({it.link for it in cluster})[:3])
    return "evt_" + sha1(base)[:10]

def pick_region_for_cluster(cluster: List[RawItem]) -> str:
    # if any item tagged cn => cn, else global
    for it in cluster:
        if it.region == "cn":
            return "cn"
    return "global"

def score_cluster(cluster: List[RawItem]) -> float:
    # simple: recency + source diversity
    now = now_bj()
    latest = cluster[0].published or now
    age_hours = (now - latest).total_seconds() / 3600.0
    rec = math.exp(-age_hours / 24.0)
    div = len({it.source for it in cluster})
    return rec * 0.8 + clamp(div / 4.0, 0.0, 0.2)

def extractive_event_card(cluster: List[RawItem], section_id: str) -> EventCard:
    rep = cluster[0]
    title_zh = to_zh(rep.title)
    # minimal "结论优先" fallback
    conclusion = f"{title_zh}（多源报道）"
    evidence = []
    if rep.summary:
        evidence.append("要点：" + shorten_cn(to_zh(rep.summary), 30, 80))
    evidence.append(f"来源数：{len(cluster)}")
    impact = "影响：待进一步确认（自动降级输出）。"
    next_step = "下一步：点击来源链接核对关键事实锚点（数字/日期/机构）。"
    sources = [(it.source, it.link) for it in cluster[:5]]
    return EventCard(
        event_id=event_id_from_cluster(cluster, section_id),
        title_zh=title_zh,
        conclusion_zh=conclusion,
        evidence_zh=evidence[:3],
        impact_zh=impact,
        next_step_zh=next_step,
        sources=sources,
        published=rep.published,
        region=pick_region_for_cluster(cluster),
        score=score_cluster(cluster),
    )

def llm_event_card(cluster: List[RawItem], section_name: str, section_id: str) -> EventCard:
    # Only 1 LLM call per cluster; if fails, fallback to extractive
    rep = cluster[0]
    title_zh = to_zh(rep.title)
    sources = [(it.source, it.link) for it in cluster[:5]]
    bullets = []
    for it in cluster[:3]:
        s = it.summary or ""
        s = strip_html(s)
        if s:
            bullets.append(f"- {it.source}: {shorten_cn(to_zh(s), 40, 90)}")
        else:
            bullets.append(f"- {it.source}: {to_zh(it.title)}")

    sys_prompt = (
        "你是资深新闻编辑与风险分析师。输出严格用于企业决策晨报，必须具体、可验证、结论优先。"
        "不要空话，不要泛泛而谈。"
    )

    user = f"""
栏目：{section_name}
事件标题：{title_zh}

已知多源要点（可能包含重复）：
{chr(10).join(bullets)}

请输出一个 JSON：
{{
  "conclusion": "一句话结论（<=28字，具体到对象/方向）",
  "evidence": ["证据要点1（含机构/数字/日期）","证据要点2", "证据要点3"],
  "impact": "影响（合规/采购/融资/产品至少选两类具体说明）",
  "next_step": "下一步动作（动词+对象+验收标准）"
}}
"""

    j = llm_json(sys_prompt, user, max_tokens=380)
    if not j:
        return extractive_event_card(cluster, section_id)

    conclusion = safe_text(j.get("conclusion"))
    evidence = j.get("evidence") if isinstance(j.get("evidence"), list) else []
    evidence = [safe_text(x) for x in evidence if safe_text(x)]
    impact = safe_text(j.get("impact"))
    next_step = safe_text(j.get("next_step"))

    # Safety fallback if too empty
    if not conclusion or len(evidence) < 1:
        return extractive_event_card(cluster, section_id)

    return EventCard(
        event_id=event_id_from_cluster(cluster, section_id),
        title_zh=title_zh,
        conclusion_zh=conclusion,
        evidence_zh=evidence[:3],
        impact_zh=impact or "影响：待进一步核实。",
        next_step_zh=next_step or "下一步：核对来源并跟踪关键阈值。",
        sources=sources,
        published=rep.published,
        region=pick_region_for_cluster(cluster),
        score=score_cluster(cluster),
    )


# -----------------------------
# Section brief (300–500 chars)
# -----------------------------

def extractive_section_brief(section_name: str, events: List[EventCard]) -> str:
    # fallback brief with concrete mentions
    lines = [f"{section_name}：今日信号以“{events[0].conclusion_zh}”为主。" if events else f"{section_name}：暂无高质量事件。"]
    if events:
        for ev in events[:4]:
            lines.append(f"• {ev.conclusion_zh}")
        lines.append("建议：优先核对带数字/日期的证据要点，并关注右侧阈值触发提示。")
    s = " ".join(lines)
    return shorten_cn(s, 180, 480)

def llm_section_brief(section_name: str, events: List[EventCard]) -> str:
    if not OPENAI_API_KEY or not _llm_budget_ok(DEFAULT_LLM_BUDGET):
        return extractive_section_brief(section_name, events)

    # build fact anchors from event evidence
    fact_lines = []
    for ev in events[:8]:
        ev_line = f"- {ev.conclusion_zh} | 证据: " + "；".join(ev.evidence_zh[:2])
        fact_lines.append(ev_line)

    sys_prompt = (
        "你是企业晨报主编。生成栏目简报（中文300-500字），必须："
        "1) 以结论开头；2) 引用事实锚点（机构/数字/日期）；"
        "3) 明确影响对象（至少两类：合规/采购/融资/产品）；"
        "4) 给出2-4条可执行动作；5) 给出2-4条监控阈值/需要持续观察的指标。"
        "不要套话。"
    )

    user = f"""
栏目：{section_name}
事件信号（结论+证据）：
{chr(10).join(fact_lines)}

输出一段中文简报，长度 300-500 字。
"""
    txt = llm_text(sys_prompt, user, max_tokens=520)
    if not txt:
        return extractive_section_brief(section_name, events)
    return shorten_cn(txt, 280, 500)


# -----------------------------
# Top brief (300–500 chars)
# -----------------------------

def extractive_top_brief(sections: List[SectionOutput], kpis: List[KPI]) -> str:
    # pick strongest conclusions
    picks = []
    for s in sections:
        if s.events:
            picks.append(s.events[0].conclusion_zh)
    picks = picks[:6]
    kpi_bits = []
    for k in kpis[:5]:
        if k.value is None:
            continue
        sign = "+" if (k.delta or 0) > 0 else ""
        d = f"{sign}{k.delta:.2f}" if k.delta is not None else "NA"
        kpi_bits.append(f"{k.name}{k.unit}:{k.value:.2f}({d})")
    s = "今日要点："
    if kpi_bits:
        s += "市场快照 " + "；".join(kpi_bits) + "。"
    if picks:
        s += " 主要事件信号：" + "；".join(picks) + "。"
    s += " 建议：优先处理制裁/合规与关键大宗价格波动相关事项，并按阈值触发进行跟踪。"
    return shorten_cn(s, 300, 500)

def llm_top_brief(sections: List[SectionOutput], kpis: List[KPI]) -> str:
    if not OPENAI_API_KEY or not _llm_budget_ok(DEFAULT_LLM_BUDGET):
        return extractive_top_brief(sections, kpis)

    event_lines = []
    for s in sections:
        for ev in s.events[:2]:
            event_lines.append(f"- [{s.name}] {ev.conclusion_zh} | 证据: {'；'.join(ev.evidence_zh[:2])}")
    event_lines = event_lines[:12]

    kpi_lines = []
    for k in kpis[:8]:
        if k.value is None:
            continue
        kpi_lines.append(f"- {k.name}: {k.value:.3f}{k.unit} (Δ {k.delta:.3f if k.delta is not None else 'NA'})")

    sys_prompt = (
        "你是企业晨报主编。输出“今日要点”中文300-500字，必须包含："
        "1) 关键结论；2) 至少3条事实锚点（机构/数字/日期）；"
        "3) 影响对象（至少两类：合规/采购/融资/产品）；"
        "4) 2-4条今日动作；5) 2-4条监控阈值/指标。"
        "不要空话，不要泛泛而谈。"
    )

    user = f"""
市场快照（KPI）：
{chr(10).join(kpi_lines) if kpi_lines else "- NA"}

事件信号（结论+证据）：
{chr(10).join(event_lines) if event_lines else "- NA"}

输出一段中文300-500字“今日要点”。
"""
    txt = llm_text(sys_prompt, user, max_tokens=520)
    if not txt:
        return extractive_top_brief(sections, kpis)
    return shorten_cn(txt, 300, 500)


# -----------------------------
# KPI computation & sparkline svg
# -----------------------------

def compute_kpi(kcfg: Dict[str, Any]) -> KPI:
    kid = safe_text(kcfg.get("id") or kcfg.get("kpi_id") or "")
    name = safe_text(kcfg.get("name") or kid)
    unit = safe_text(kcfg.get("unit") or "")
    typ = safe_text(kcfg.get("type") or "fred").lower()

    series = safe_text(kcfg.get("series") or kcfg.get("series_id") or kid)

    values: List[float] = []
    if typ == "fred" and series:
        pts = int(kcfg.get("points") or 30)
        data = fetch_fred_csv(series, pts)
        values = [v for _, v in data]

    if not values:
        return KPI(kpi_id=kid, name=name, unit=unit, value=None, delta=None, spark=[], status="na")

    value = values[-1]
    prev = values[-2] if len(values) >= 2 else values[-1]
    delta = value - prev
    if abs(delta) < 1e-9:
        status = "flat"
    else:
        status = "up" if delta > 0 else "down"

    return KPI(kpi_id=kid, name=name, unit=unit, value=value, delta=delta, spark=values[-20:], status=status)

def spark_svg(values: List[float], width: int = 120, height: int = 34) -> str:
    if not values:
        return ""
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = i * (width - 2) / max(1, (n - 1)) + 1
        y = (height - 2) - (v - vmin) * (height - 2) / (vmax - vmin) + 1
        pts.append((x, y))
    d = "M " + " L ".join([f"{x:.2f},{y:.2f}" for x, y in pts])
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true"><path d="{d}" fill="none" stroke="currentColor" stroke-width="2" /></svg>'


# -----------------------------
# Section builder
# -----------------------------

def build_section(section_cfg: Dict[str, Any], limit_raw: int, items_per_section: int, cluster_threshold: float) -> SectionOutput:
    sid = safe_text(section_cfg.get("id") or section_cfg.get("section_id") or "")
    name = safe_text(section_cfg.get("name") or sid)

    print(f"[digest] section start: {sid} ({name})", flush=True)

    sources_cfg = section_cfg.get("sources") or []
    # Accept shorthand: {"rss": [..]}
    if not sources_cfg and "rss" in section_cfg:
        sources_cfg = [{"type": "rss", "name": "RSS", "url": u} for u in section_cfg["rss"]]

    raw: List[RawItem] = []
    # distribute raw limit across sources
    per_src = max(5, limit_raw // max(1, len(sources_cfg)))
    for s in sources_cfg:
        stype = safe_text(s.get("type") or "rss").lower()
        sname = safe_text(s.get("name") or "Source")
        url = safe_text(s.get("url") or "")
        if not url:
            continue
        if stype == "rss":
            got = fetch_rss(url, sname, per_src)
        else:
            got = fetch_rss(url, sname, per_src)
        for it in got:
            it.section_id = sid
            it.region = classify_region_rule(it.title + " " + it.summary)
        raw.extend(got)

    # de-dup by link
    seen = set()
    uniq: List[RawItem] = []
    for it in raw:
        if it.link in seen:
            continue
        seen.add(it.link)
        uniq.append(it)

    # sort by time desc (unknown at end)
    uniq.sort(key=lambda x: x.published or dt.datetime(1970, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=8))), reverse=True)

    # cluster
    clusters = cluster_items(uniq[:limit_raw], threshold=cluster_threshold)

    # event cards: LLM for top clusters; fallback for rest to keep total LLM calls bounded
    events: List[EventCard] = []
    for c in clusters:
        if len(events) >= items_per_section:
            break
        # LLM budget: use LLM for first few clusters, then fallback
        use_llm = _llm_budget_ok(DEFAULT_LLM_BUDGET) and len(events) < max(6, items_per_section // 2)
        ev = llm_event_card(c, name, sid) if use_llm else extractive_event_card(c, sid)
        events.append(ev)

    # sort by score
    events.sort(key=lambda e: e.score, reverse=True)

    # section brief (300-500)
    brief = llm_section_brief(name, events) if events else f"{name}：今日未抓取到足够高质量条目。"

    print(f"[digest] section done: {sid} events={len(events)}", flush=True)

    return SectionOutput(section_id=sid, name=name, brief_zh=brief, events=events)


# -----------------------------
# Views (all / cn / global-ex-cn)
# -----------------------------

def filter_sections(sections: List[SectionOutput], mode: str) -> List[SectionOutput]:
    if mode == "all":
        return sections
    out: List[SectionOutput] = []
    for s in sections:
        evs = [e for e in s.events if e.region == mode]
        out.append(SectionOutput(section_id=s.section_id, name=s.name, brief_zh=s.brief_zh, events=evs))
    return out


# -----------------------------
# Template render
# -----------------------------

def render_html(template_path: str, context: Dict[str, Any]) -> str:
    tpl_dir = os.path.dirname(os.path.abspath(template_path)) or "."
    tpl_name = os.path.basename(template_path)
    env = Environment(
        loader=FileSystemLoader(tpl_dir),
        autoescape=False,
        undefined=Undefined,  # missing vars won't crash
    )
    template = env.get_template(tpl_name)
    return template.render(**context)


# -----------------------------
# Config loading with defaults
# -----------------------------

def default_config() -> Dict[str, Any]:
    # A minimal fallback config using Google News RSS queries (global, authoritative-ish)
    # You should maintain your own digest_config_v15.json in repo.
    def gnews(q: str) -> str:
        # hl=en-US&gl=US&ceid=US:en
        import urllib.parse
        return "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en"

    return {
        "site_title": "Ben的每日资讯简报",
        "sections": [
            {"id": "macro_markets", "name": "宏观与市场", "sources": [{"type": "rss", "name": "Google News", "url": gnews("inflation OR rates OR treasury OR dollar OR stock market")}], "items_per_section": 15},
            {"id": "geopolitics_sanctions", "name": "地缘政治与制裁", "sources": [{"type": "rss", "name": "Google News", "url": gnews("sanctions OR OFAC OR export controls OR geopolitics")}], "items_per_section": 15},
            {"id": "ai_compute", "name": "AI算力与基础设施", "sources": [{"type": "rss", "name": "Google News", "url": gnews("NVIDIA OR AI chips OR data center power OR hyperscaler capex")}], "items_per_section": 15},
            {"id": "commodities_metals", "name": "大宗与金属", "sources": [{"type": "rss", "name": "Google News", "url": gnews("copper price OR aluminum price OR nickel price OR LME")}], "items_per_section": 15},
            {"id": "carbon_cbam", "name": "碳与CBAM", "sources": [{"type": "rss", "name": "Google News", "url": gnews("CBAM OR carbon credit OR Verra OR EU ETS")}], "items_per_section": 15},
            {"id": "sea_supplychain", "name": "东南亚供应链", "sources": [{"type": "rss", "name": "Google News", "url": gnews("Vietnam supply chain OR Thailand manufacturing OR Malaysia semiconductor")}], "items_per_section": 15},
        ],
        "kpis": [
            {"id": "DGS10", "name": "美国10Y", "unit": "%", "type": "fred", "series": "DGS10", "points": 30},
            {"id": "DTB3", "name": "美国3M", "unit": "%", "type": "fred", "series": "DTB3", "points": 30},
        ],
    }

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[digest] config not found, using defaults: {path}", flush=True)
        return default_config()
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", default="index.html")
    parser.add_argument("--date", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit_raw", type=int, default=25)
    parser.add_argument("--cluster_threshold", type=float, default=0.82)
    parser.add_argument("--items_per_section", type=int, default=0)  # optional; config-driven if 0
    args = parser.parse_args()

    if args.model:
        global OPENAI_MODEL
        OPENAI_MODEL = args.model.strip()

    _load_cache()

    bj_now = now_bj()
    run_date = bj_now.date()
    if args.date:
        try:
            run_date = dt.date.fromisoformat(args.date)
        except Exception:
            pass

    print(f"[digest] date={run_date} model={OPENAI_MODEL} limit_raw={args.limit_raw} thr={args.cluster_threshold} budget={DEFAULT_LLM_BUDGET}", flush=True)

    cfg = load_config(args.config)
    site_title = safe_text(cfg.get("site_title") or "Ben的每日资讯简报")

    # KPIs
    kpis_cfg = cfg.get("kpis") or []
    kpis: List[KPI] = []
    print(f"[digest] kpi start: {len(kpis_cfg)}", flush=True)
    for kc in kpis_cfg:
        try:
            k = compute_kpi(kc)
            kpis.append(k)
        except Exception as e:
            print(f"[digest] kpi fail: {kc} | {e}", flush=True)
    print(f"[digest] kpi done: {len(kpis)}", flush=True)

    # Sections
    sections_cfg = cfg.get("sections") or []
    sections: List[SectionOutput] = []
    print(f"[digest] sections start: {len(sections_cfg)}", flush=True)

    for sc in sections_cfg:
        items_per = int(args.items_per_section) if args.items_per_section > 0 else int(sc.get("items_per_section") or 15)
        try:
            sec = build_section(sc, limit_raw=int(args.limit_raw), items_per_section=items_per, cluster_threshold=float(args.cluster_threshold))
            sections.append(sec)
        except Exception as e:
            sid = safe_text(sc.get("id") or sc.get("section_id") or "unknown")
            print(f"[digest] section fail: {sid} | {type(e).__name__}: {e}", flush=True)
            sections.append(SectionOutput(section_id=sid, name=safe_text(sc.get("name") or sid), brief_zh="该栏目生成失败（已降级）。", events=[]))

    print(f"[digest] sections done: {len(sections)}", flush=True)

    # Top brief (all view based)
    top_brief_all = llm_top_brief(sections, kpis)
    # Views filter
    sections_cn = filter_sections(sections, "cn")
    sections_global = filter_sections(sections, "global")

    top_brief_cn = llm_top_brief(sections_cn, kpis) if any(s.events for s in sections_cn) else top_brief_all
    top_brief_global = llm_top_brief(sections_global, kpis) if any(s.events for s in sections_global) else top_brief_all

    # Prepare template context (provide rich keys; template can ignore extras)
    def sections_to_dict(secs: List[SectionOutput]) -> List[Dict[str, Any]]:
        out = []
        for s in secs:
            out.append({
                "id": s.section_id,
                "name": s.name,
                "brief_zh": s.brief_zh,
                "events": [
                    {
                        "event_id": e.event_id,
                        "title_zh": e.title_zh,
                        "conclusion_zh": e.conclusion_zh,
                        "evidence_zh": e.evidence_zh,
                        "impact_zh": e.impact_zh,
                        "next_step_zh": e.next_step_zh,
                        "sources": [{"name": n, "url": u} for n, u in e.sources],
                        "published": e.published.isoformat() if e.published else "",
                        "region": e.region,
                        "score": e.score,
                    }
                    for e in s.events
                ]
            })
        return out

    def kpis_to_dict(ks: List[KPI]) -> List[Dict[str, Any]]:
        out = []
        for k in ks:
            out.append({
                "id": k.kpi_id,
                "name": k.name,
                "unit": k.unit,
                "value": k.value,
                "delta": k.delta,
                "status": k.status,  # up/down/flat/na
                "spark_values": k.spark,
                "spark_svg": spark_svg(k.spark) if k.spark else "",
            })
        return out

    views = {
        "all": {
            "top_brief_zh": top_brief_all,
            "sections": sections_to_dict(sections),
        },
        "cn": {
            "top_brief_zh": top_brief_cn,
            "sections": sections_to_dict(sections_cn),
        },
        "global": {
            "top_brief_zh": top_brief_global,
            "sections": sections_to_dict(sections_global),
        }
    }

    context = {
        "site_title": site_title,
        "date": to_date_str(run_date),
        "generated_at": now_bj().strftime("%Y-%m-%d %H:%M (北京时间)"),
        "llm_model": OPENAI_MODEL,
        "llm_calls_used": _llm_calls_used,
        "kpis": kpis_to_dict(kpis),
        "views": views,
        # convenience top-level (for templates that expect these keys)
        "top_brief_zh": top_brief_all,
        "sections": sections_to_dict(sections),
    }

    print("[digest] render html", flush=True)
    html = render_html(args.template, context)

    # write output
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    _save_cache()
    print(f"[digest] done: wrote {out_path} | llm_calls={_llm_calls_used}", flush=True)


if __name__ == "__main__":
    main()
