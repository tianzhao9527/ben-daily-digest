#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v18.py

V18 单页静态简报生成器（结论优先 + 三视图预渲染）
- 生成端拉取/筛选/聚类/翻译/总结 -> 输出静态 HTML（index.html）
- 页面端仅做：视图切换（全部/中国相关/全球不含中国）、搜索、展开/收起、反馈（LocalStorage）
- 不做任何 client-side 抓取（打开页面不会再请求新闻源/数据源）

数据源（默认）
- 新闻：Google News RSS（按栏目关键词；可限制来源白名单；拒绝 .cn 域名）
- 论文：arXiv API
- 数据：FRED CSV（无需 key）

依赖：
  pip install requests feedparser jinja2
（GitHub Actions 会自动安装 requirements.txt）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import hashlib
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, quote_plus
from typing import Any

import requests
import feedparser
from jinja2 import Template

import time, sys
import faulthandler, signal

# 允许 workflow 发送 SIGUSR1 打印所有线程堆栈（定位卡点）
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
print("[digest] boot", flush=True)

import requests
_real_get = requests.get

def _get(*args, **kwargs):
    kwargs.setdefault("timeout", (5, 20))  # connect 5s, read 20s
    return _real_get(*args, **kwargs)
requests.get = _get

BJT = timezone(timedelta(hours=8))

# -----------------------------
# Time / Text Utils
# -----------------------------
def now_bjt() -> datetime:
    return datetime.now(tz=BJT)

def now_bjt_str() -> str:
    return now_bjt().strftime("%Y-%m-%d %H:%M")

def date_bjt_str() -> str:
    return now_bjt().strftime("%Y-%m-%d")

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\u3000]+", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\s\-\.\:/]", "", s)
    return s

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def safe_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

# -----------------------------
# OpenAI (optional)
# -----------------------------
def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

def openai_chat(messages: list[dict], model: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    model = os.environ.get("OPENAI_MODEL") or model
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def llm_json(system: str, user: str, model: str, max_tokens: int = 900, temperature: float = 0.2) -> dict:
    """Return JSON dict; if parsing fails, return {}."""
    if not llm_available():
        return {}
    try:
        txt = openai_chat(
            [
                {"role": "system", "content": system + "\n只输出 JSON，不要输出任何解释文字。"},
                {"role": "user", "content": user},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception:
        return {}

# -----------------------------
# Fetchers
# -----------------------------
def fetch_google_news_rss(query: str, lang: str = "en", region: str = "US", limit: int = 80) -> list[dict]:
    q = quote_plus(query)
    ceid = f"{region}:{lang}"
    url = f"https://news.google.com/rss/search?q={q}&hl={lang}-{region}&gl={region}&ceid={ceid}"
    feed = feedparser.parse(url)
    items: list[dict] = []
    for e in getattr(feed, "entries", [])[:limit]:
        link = getattr(e, "link", "") or ""
        title = getattr(e, "title", "") or ""
        published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
        summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
        source = ""
        try:
            source = getattr(getattr(e, "source", None), "title", "") or ""
        except Exception:
            source = ""
        items.append({
            "title": title,
            "url": link,
            "published": published,
            "summary": re.sub(r"<[^>]+>", "", summary).strip(),
            "source": source or safe_domain(link) or "Google News",
        })
    return items

def fetch_rss(url: str, limit: int = 60, source: str | None = None) -> list[dict]:
    feed = feedparser.parse(url)
    items: list[dict] = []
    for e in getattr(feed, "entries", [])[:limit]:
        link = getattr(e, "link", "") or ""
        title = getattr(e, "title", "") or ""
        published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
        summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
        src = source or getattr(feed, "feed", {}).get("title", "") or safe_domain(url) or "RSS"
        items.append({
            "title": title,
            "url": link,
            "published": published,
            "summary": re.sub(r"<[^>]+>", "", summary).strip(),
            "source": src,
        })
    return items

def fetch_arxiv(search_query: str, limit: int = 30) -> list[dict]:
    # arXiv API: Atom feed
    q = quote_plus(search_query)
    url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    items: list[dict] = []
    for e in getattr(feed, "entries", [])[:limit]:
        link = ""
        for l in getattr(e, "links", []) or []:
            if getattr(l, "rel", "") == "alternate":
                link = getattr(l, "href", "") or link
        title = (getattr(e, "title", "") or "").replace("\n", " ").strip()
        summary = (getattr(e, "summary", "") or "").replace("\n", " ").strip()
        published = getattr(e, "published", "") or ""
        items.append({
            "title": title,
            "url": link,
            "published": published,
            "summary": summary,
            "source": "arXiv",
        })
    return items

# -----------------------------
# Region classification
# -----------------------------
CN_KW = [
    "china","chinese","beijing","shanghai","shenzhen","hong kong","taiwan","cny","rmb","yuan",
    "huawei","byd","tencent","alibaba","xiaomi","tiktok","shein","temu","pinduoduo","sinopec","sinochem","ant group",
    "hangzhou","jiangsu","zhejiang","shandong","pla","prc"
]

def classify_region(title: str, model: str = "gpt-4o-mini") -> str:
    t = norm_text(title)
    if not t:
        return "global"
    if llm_available():
        js = llm_json(
            system="你是新闻分类器。",
            user=f"判断标题是否与中国相关（政策/公司/供应链/市场/地缘）。只输出 JSON：{{\"region\":\"cn\"或\"global\"}}。\n标题：{title}",
            model=model,
            max_tokens=60,
        )
        if js.get("region") in ("cn", "global"):
            return js["region"]
    for k in CN_KW:
        if k in t:
            return "cn"
    if any(x in title for x in ["中国","中方","北京","上海","深圳","香港","台湾","人民币","央行","发改委","工信部"]):
        return "cn"
    return "global"

# -----------------------------
# Clustering
# -----------------------------
def tokenize_for_cluster(s: str) -> set[str]:
    s = norm_text(s)
    if not s:
        return set()
    parts = re.split(r"[\s\-/:\.]+", s)
    toks: list[str] = []
    for w in parts:
        w = w.strip()
        if not w:
            continue
        if re.fullmatch(r"[a-z0-9]{2,}", w):
            toks.append(w)
        else:
            # crude CJK bigrams
            if len(w) >= 2:
                toks.extend([w[i:i+2] for i in range(0, len(w)-1)])
            else:
                toks.append(w)
    return set(toks)

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def cluster_items(items: list[dict], threshold: float = 0.42, max_clusters: int = 80) -> list[dict]:
    clusters: list[dict] = []
    for it in items:
        tset = tokenize_for_cluster(it.get("title",""))
        if not tset:
            continue
        best_i, best_s = -1, 0.0
        for i, c in enumerate(clusters):
            s = jaccard(tset, c["tset"])
            if s > best_s:
                best_s = s
                best_i = i
        if best_s >= threshold and best_i >= 0:
            clusters[best_i]["items"].append(it)
            # keep representative title as the shortest (usually cleaner)
            if len(it.get("title","")) < len(clusters[best_i].get("title","") or ""):
                clusters[best_i]["title"] = it.get("title","")
        else:
            clusters.append({"title": it.get("title",""), "items":[it], "tset": tset})
        if len(clusters) >= max_clusters:
            break
    # sort: clusters with more sources first, then recency (if available)
    clusters.sort(key=lambda c: (len(c["items"]), c["items"][0].get("published","")), reverse=True)
    for c in clusters:
        c.pop("tset", None)
        c["id"] = sha1((c.get("title","") + "|" + (c["items"][0].get("url",""))))[:12]
    return clusters

# -----------------------------
# KPI / FRED
# -----------------------------
def fetch_fred_series(series_id: str, limit: int = 60) -> list[float]:
    # FRED provides CSV download without key
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    vals: list[float] = []
    lines = r.text.strip().splitlines()
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        v = parts[1].strip()
        if v in (".", ""):
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    return vals[-limit:] if vals else []

def spark_svg(values: list[float], direction: str | None = None) -> str:
    """Inline SVG sparkline; color via CSS currentColor on wrapper (.up/.down)."""
    if not values or len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        vmax = vmin + 1e-9
    W, H = 110.0, 44.0
    pts = []
    for i, v in enumerate(values):
        x = 2 + (W - 4) * (i / (len(values) - 1))
        y = 6 + (H - 12) * (1 - (v - vmin) / (vmax - vmin))
        pts.append((x, y))
    d = "M " + " L ".join([f"{x:.2f},{y:.2f}" for x, y in pts])
    return f'<svg viewBox="0 0 {W:.0f} {H:.0f}" class="spark" aria-hidden="true"><path d="{d}" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>'

def compute_delta(values: list[float]) -> tuple[float|None, float|None]:
    if not values or len(values) < 2:
        return (None, None)
    last = values[-1]
    prev = values[-2]
    if prev == 0:
        return (last-prev, None)
    return (last-prev, (last-prev)/abs(prev)*100.0)

def build_kpis(cfg: dict) -> list[dict]:
    out: list[dict] = []
    for k in cfg.get("kpis", []):
        sid = k.get("series")
        if not sid:
            continue
        try:
            vals = fetch_fred_series(sid, limit=int(k.get("lookback", 40)))
        except Exception:
            vals = []
        last = vals[-1] if vals else None
        d_abs, d_pct = compute_delta(vals)
        direction = None
        if d_abs is not None:
            direction = "up" if d_abs > 0 else ("down" if d_abs < 0 else "flat")
        out.append({
            "id": sid,
            "title": k.get("title") or sid,
            "unit": k.get("unit") or "",
            "value": last,
            "delta_abs": d_abs,
            "delta_pct": d_pct,
            "spark": spark_svg(vals, direction=direction),
            "direction": direction,
            "source": "FRED",
        })
    return out

def eval_rule(value: float | None, rule: dict) -> bool:
    if value is None:
        return False
    op = rule.get("op")
    thr = rule.get("value")
    try:
        thr = float(thr)
    except Exception:
        return False
    if op == ">":
        return value > thr
    if op == ">=":
        return value >= thr
    if op == "<":
        return value < thr
    if op == "<=":
        return value <= thr
    if op == "abs>":
        return abs(value) > thr
    return False

def build_alerts(cfg: dict, kpis: list[dict]) -> list[dict]:
    rules = cfg.get("alerts", [])
    kmap = {k["id"]: k for k in kpis}
    fired: list[dict] = []
    for r in rules:
        sid = r.get("series")
        if not sid or sid not in kmap:
            continue
        kv = kmap[sid]
        if eval_rule(kv.get("value"), r):
            fired.append({
                "series": sid,
                "title": kv.get("title"),
                "value": kv.get("value"),
                "unit": kv.get("unit"),
                "op": r.get("op"),
                "threshold": r.get("value"),
                "message": r.get("message",""),
                "severity": r.get("severity","info"),
            })
    return fired[:12]

# -----------------------------
# Translation helper
# -----------------------------
def to_zh(text: str, model: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if not llm_available():
        return text  # fallback
    js = llm_json(
        system="你是专业译者与编辑。",
        user=f"把下面内容翻译成简体中文，要求准确、简洁、保留专有名词与数字。只输出 JSON：{{\"zh\":\"...\"}}\n\n{text}",
        model=model,
        max_tokens=700,
        temperature=0.1,
    )
    zh = (js.get("zh") or "").strip()
    return zh or text

# -----------------------------
# Event cards (结论优先)
# -----------------------------
EVENT_SCHEMA = {
    "headline_zh": "string (<=40字，事件标题，中文)",
    "conclusion": "string (<=50字，一句话结论：发生了什么+对谁重要)",
    "evidence": ["string (2-3条，带来源名的事实锚点；尽量包含数字/机构/时间)"],
    "impact": ["string (1-2条，对合规/采购/融资/产品的影响)"],
    "next": ["string (1-2条，下一步动作或要看的指标/阈值)"],
    "tags": ["string (<=6个，中文标签)"],
}

def event_card_from_cluster(cluster: dict, model: str, max_sources: int = 5) -> dict:
    items = (cluster.get("items") or [])[:max_sources]
    # build richer context
    src_lines = []
    for it in items:
        src_lines.append(
            f"- 来源:{it.get('source','')} | 时间:{it.get('published','')[:25]} | 标题:{it.get('title','')}\n"
            f"  摘要:{(it.get('summary','') or '')[:240]}\n"
            f"  链接:{it.get('url','')}"
        )
    ctx = "\n".join(src_lines)
    if llm_available():
        js = llm_json(
            system="你是资深研究助理，擅长把同一事件的多来源报道合并成“结论优先”的事件卡。必须具体，避免空话。",
            user=(
                "请基于多来源材料，输出一张事件卡（结论优先）。\n"
                "硬性要求：\n"
                "1) evidence 至少 2 条，且尽量包含数字/机构/时间；没有就明确写‘未给出具体数字’。\n"
                "2) conclusion 不要抽象判断，必须落到具体事件。\n"
                "3) next 必须是可执行动作或监控指标/阈值。\n"
                f"输出严格 JSON：{json.dumps(EVENT_SCHEMA, ensure_ascii=False)}\n\n"
                f"材料：\n{ctx}\n"
            ),
            model=model,
            max_tokens=900,
            temperature=0.2,
        )
    else:
        js = {}
    # sanitize
    out = {k: js.get(k) for k in EVENT_SCHEMA.keys()}
    out["headline_zh"] = (out.get("headline_zh") or to_zh(cluster.get("title",""), model)).strip()
    out["conclusion"] = (out.get("conclusion") or "").strip()
    for k in ("evidence","impact","next","tags"):
        v = out.get(k) or []
        if isinstance(v, str):
            v = [v]
        out[k] = [str(x).strip() for x in v if str(x).strip()][: (3 if k=="evidence" else 2 if k in ("impact","next") else 6)]
    out["sources"] = [{
        "name": it.get("source",""),
        "title": to_zh(it.get("title",""), model) if llm_available() else it.get("title",""),
        "url": it.get("url",""),
        "published": it.get("published",""),
    } for it in items]
    out["id"] = cluster.get("id")
    out["region"] = classify_region(cluster.get("title",""), model=model)  # cluster-level region
    return out

# -----------------------------
# Section briefs (300-500字 + 结构)
# -----------------------------
SECTION_SCHEMA = {
    "summary": "string (300-500字，中文段落；必须具体，避免口号)",
    "one_liner": "string (<=30字，结论一句话)",
    "facts": ["string (3-5条，事实锚点，尽量含数字/机构/时间)"],
    "impacts": ["string (2-3条，影响对象：合规/采购/融资/产品/供应链)"],
    "actions": ["string (2-4条，可执行动作，含验收标准)"],
    "watch": ["string (2-4条，监控指标/阈值；尽量引用右侧KPI或配置阈值)"],
}

def section_brief(section_name: str, cards: list[dict], kpis: list[dict], alerts: list[dict], model: str, target_chars=(300,500)) -> dict:
    # Provide top evidence snippets to ground
    lines = []
    for c in cards[:8]:
        ev = "; ".join((c.get("evidence") or [])[:2])
        lines.append(f"- {c.get('headline_zh','')}: {c.get('conclusion','')} | 证据: {ev}")
    kpi_lines = []
    for k in kpis[:10]:
        if k.get("value") is None:
            continue
        d = k.get("delta_pct")
        dstr = "" if d is None else f"{d:+.2f}%"
        kpi_lines.append(f"- {k.get('title')}({k.get('id')}): {k.get('value')} {k.get('unit')} {dstr}")
    alert_lines = []
    for a in alerts[:8]:
        alert_lines.append(f"- {a.get('title')} 触发 {a.get('op')}{a.get('threshold')}: {a.get('message')}")
    ctx = "\n".join(lines)
    if llm_available():
        schema = SECTION_SCHEMA.copy()
        schema["summary"] = f"string ({target_chars[0]}-{target_chars[1]}字)"
        js = llm_json(
            system="你是CEO的栏目情报官，目标是让读者在30秒内抓住本栏目最关键的变化，并给出可执行动作。",
            user=(
                f"请为栏目《{section_name}》生成栏目决策卡与简报。\n"
                f"要求：summary 必须 {target_chars[0]}–{target_chars[1]} 字；必须引用材料中的事实锚点（数字/机构/时间/实体），不要空话。\n"
                "facts 至少3条；actions 至少2条；watch 至少2条。\n"
                f"输出严格 JSON：{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"材料（事件卡摘要）：\n{ctx}\n\n"
                f"今日KPI快照：\n{chr(10).join(kpi_lines[:12])}\n\n"
                f"已触发提示（如有）：\n{chr(10).join(alert_lines) if alert_lines else '无'}\n"
            ),
            model=model,
            max_tokens=1100,
            temperature=0.2,
        )
    else:
        js = {}
    out = {}
    out["summary"] = str(js.get("summary") or "").strip()
    out["one_liner"] = str(js.get("one_liner") or "").strip()
    for k in ("facts","impacts","actions","watch"):
        v = js.get(k) or []
        if isinstance(v, str):
            v = [v]
        # trim
        lim = 5 if k=="facts" else 3 if k=="impacts" else 4
        out[k] = [str(x).strip() for x in v if str(x).strip()][:lim]
    # fallback if missing
    if not out["summary"]:
        out["summary"] = (f"本栏目今日共聚类 {len(cards)} 个事件。建议优先阅读前3个事件卡，并结合右侧KPI与阈值进行判断。"
                          f"（未配置 OPENAI_API_KEY 时为占位文本。）")
    if not out["one_liner"]:
        out["one_liner"] = f"{section_name}：关注核心变化与可执行动作"
    return out

# -----------------------------
# Top brief (300-500字 + 结构)
# -----------------------------
TOP_SCHEMA = {
    "summary": "string (300-500字，今日要点正文；必须具体)",
    "one_liner": "string (<=30字，今天最重要的判断)",
    "facts": ["string (3-6条，跨栏目事实锚点)"],
    "actions": ["string (3-5条，今天的动作/决策)"],
    "watch": ["string (3-5条，监控指标/阈值)"],
}

def top_brief(view_name: str, sections_payload: list[dict], kpis: list[dict], alerts: list[dict], model: str, target_chars=(300,500)) -> dict:
    # Grounding: take first 2 events per section
    lines = []
    for s in sections_payload:
        for c in (s.get("cards") or [])[:2]:
            ev = "; ".join((c.get("evidence") or [])[:2])
            lines.append(f"- [{s.get('name')}] {c.get('headline_zh')}: {c.get('conclusion')} | 证据:{ev}")
    kpi_lines = []
    for k in kpis[:12]:
        if k.get("value") is None:
            continue
        d = k.get("delta_pct")
        dstr = "" if d is None else f"{d:+.2f}%"
        kpi_lines.append(f"- {k.get('title')}({k.get('id')}): {k.get('value')} {k.get('unit')} {dstr}")
    alert_lines = []
    for a in alerts[:10]:
        alert_lines.append(f"- {a.get('title')} 触发 {a.get('op')}{a.get('threshold')}：{a.get('message')}")
    if llm_available():
        schema = TOP_SCHEMA.copy()
        schema["summary"] = f"string ({target_chars[0]}-{target_chars[1]}字)"
        js = llm_json(
            system="你是CEO的每日情报官。你必须‘结论优先’，并用事实锚点支撑，不要空话。",
            user=(
                f"请生成《Top 今日要点》（视图：{view_name}）。\n"
                f"要求：summary 必须 {target_chars[0]}–{target_chars[1]} 字；必须引用至少 3 条事实锚点（数字/机构/时间/实体）；actions 给可执行动作；watch 给阈值。\n"
                f"输出严格 JSON：{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"材料（跨栏目事件卡摘要）：\n{chr(10).join(lines[:30])}\n\n"
                f"今日KPI快照：\n{chr(10).join(kpi_lines)}\n\n"
                f"已触发提示（如有）：\n{chr(10).join(alert_lines) if alert_lines else '无'}\n"
            ),
            model=model,
            max_tokens=1300,
            temperature=0.2,
        )
    else:
        js = {}
    out = {}
    out["summary"] = str(js.get("summary") or "").strip()
    out["one_liner"] = str(js.get("one_liner") or "").strip()
    for k in ("facts","actions","watch"):
        v = js.get(k) or []
        if isinstance(v, str):
            v = [v]
        lim = 6 if k=="facts" else 5
        out[k] = [str(x).strip() for x in v if str(x).strip()][:lim]
    if not out["summary"]:
        out["summary"] = "（占位）未配置 OPENAI_API_KEY，Top 今日要点无法生成。请在 GitHub Secrets 中设置 OPENAI_API_KEY。"
    if not out["one_liner"]:
        out["one_liner"] = "今天：先看触发阈值，再按栏目快速决策"
    return out

# -----------------------------
# Build sections (raw -> clusters -> cards)
# -----------------------------
def apply_source_filters(items: list[dict], cfg: dict) -> list[dict]:
    deny_tlds = cfg.get("deny_tlds", [".cn"])
    deny_domains = set([d.lower() for d in cfg.get("deny_domains", [])])
    allow_domains = set([d.lower() for d in cfg.get("allow_domains", [])])  # optional
    out = []
    for it in items:
        dom = safe_domain(it.get("url",""))
        if any(dom.endswith(tld) for tld in deny_tlds):
            continue
        if dom in deny_domains:
            continue
        if allow_domains and (dom not in allow_domains):
            # if allowlist enabled, enforce it
            continue
        out.append(it)
    return out

def section_fetch(section: dict, items_per_section: int) -> list[dict]:
    raw: list[dict] = []
    # 1) google news queries
    for q in section.get("queries", []):
        raw.extend(fetch_google_news_rss(q, lang=section.get("lang","en"), region=section.get("region","US"), limit=max(60, items_per_section*6)))
    # 2) extra rss feeds
    for f in section.get("feeds", []):
        url = f.get("url") if isinstance(f, dict) else str(f)
        src = f.get("source") if isinstance(f, dict) else None
        raw.extend(fetch_rss(url, limit=max(30, items_per_section*4), source=src))
    # 3) arxiv
    if section.get("type") == "arxiv":
        raw = fetch_arxiv(section.get("arxiv_query","cat:cs.LG"), limit=max(30, items_per_section*3))
    # de-dup by url
    seen = set()
    dedup = []
    for it in raw:
        u = it.get("url","")
        if not u or u in seen:
            continue
        seen.add(u)
        dedup.append(it)
    return dedup

def build_section(section_cfg: dict, cfg: dict, args) -> dict:
    items_per_section = int(cfg.get("items_per_section", 15))
    raw = section_fetch(section_cfg, items_per_section=items_per_section)
    raw = apply_source_filters(raw, cfg)
    # classify region
    for it in raw:
        it["region"] = classify_region(it.get("title",""), model=args.model)
    # cluster
    clusters = cluster_items(raw, threshold=float(cfg.get("cluster_threshold", 0.42)), max_clusters=int(cfg.get("limit_raw", 90)))
    # cards
    cards = [event_card_from_cluster(c, model=args.model, max_sources=int(cfg.get("max_sources", 5))) for c in clusters]
    # limit
    cards = cards[:items_per_section]
    return {
        "id": section_cfg.get("id") or sha1(section_cfg.get("name",""))[:10],
        "name": section_cfg.get("name",""),
        "cards": cards,
    }

def split_view_sections(sections: list[dict], view: str) -> list[dict]:
    if view == "all":
        return sections
    out = []
    for s in sections:
        cards = s.get("cards") or []
        if view == "cn":
            filt = [c for c in cards if c.get("region") == "cn"]
        else:  # global
            filt = [c for c in cards if c.get("region") == "global"]
        out.append({**s, "cards": filt})
    return out

def nav_counts(sections: list[dict]) -> list[dict]:
    return [{"id": s["id"], "name": s["name"], "count": len(s.get("cards") or [])} for s in sections]

# -----------------------------
# Rendering
# -----------------------------
def render_html(template_path: str, digest: dict, out_path: str) -> None:
    tpl = Template(open(template_path, "r", encoding="utf-8").read())
    html = tpl.render(**digest)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# -----------------------------
# Default config (if missing)
# -----------------------------
DEFAULT_CONFIG = {
  "items_per_section": 15,
  "cluster_threshold": 0.42,
  "limit_raw": 90,
  "max_sources": 5,
  "deny_tlds": [".cn"],
  "sections": [
    {
      "id": "macro",
      "name": "宏观 / 市场（利率·汇率·风险偏好）",
      "queries": [
        "US Treasury yields outlook Fed pricing inflation jobs",
        "USD CNY FX policy capital flows emerging markets risk sentiment",
        "global equity volatility VIX credit spreads"
      ]
    },
    {
      "id": "sanctions",
      "name": "地缘政治 / 制裁 / 合规（OFAC·EU）",
      "queries": [
        "OFAC sanctions update enforcement action guidance",
        "EU sanctions package Russia export controls compliance",
        "US export controls semiconductor AI chips restrictions"
      ],
      "feeds": [
        {"source":"OFAC", "url":"https://home.treasury.gov/ofac/press-releases/rss.xml"},
        {"source":"EU Council", "url":"https://www.consilium.europa.eu/en/press/press-releases/rss/"}
      ]
    },
    {
      "id": "compute",
      "name": "AI 计算基础设施（算力·电力·冷却）",
      "queries": [
        "data center power demand grid connection GPU supply chain HBM",
        "AI chip export controls cloud GPU pricing capacity",
        "liquid cooling data center PUE regulation"
      ]
    },
    {
      "id": "metals",
      "name": "大宗 / 金属（LME·供需·库存）",
      "queries": [
        "LME copper aluminium nickel zinc inventory premium spread",
        "scrap copper trade flows sanctions freight",
        "energy transition metals demand supply"
      ]
    },
    {
      "id": "carbon",
      "name": "碳 / CBAM / 碳市场（Verra·EU）",
      "queries": [
        "EU CBAM guidance reporting requirements transition period",
        "Verra methodology update registry integrity",
        "carbon credit quality additionality audit"
      ],
      "feeds": [
        {"source":"European Commission", "url":"https://ec.europa.eu/commission/presscorner/api/rss?language=en&format=rss"}
      ]
    },
    {
      "id": "sea",
      "name": "东南亚供应链（制造·转移·关键矿产）",
      "queries": [
        "Vietnam manufacturing investment supply chain relocation",
        "Indonesia nickel export policy smelter investment",
        "Thailand Malaysia electronics supply chain incentives"
      ]
    },
    {
      "id": "frontier",
      "name": "前沿学科雷达（量子·聚变·材料·生物）",
      "queries": [
        "quantum computing error correction breakthrough",
        "fusion energy high-temperature superconducting magnets",
        "new materials battery solid-state electrolyte breakthrough",
        "synthetic biology biomanufacturing platform"
      ]
    },
    {
      "id": "arxiv",
      "name": "研究雷达（arXiv）",
      "type": "arxiv",
      "arxiv_query": "cat:cs.LG OR cat:cs.AI OR cat:cs.CL"
    }
  ],
  "kpis": [
    {"series":"DGS10","title":"美债10Y","unit":"%","lookback":40},
    {"series":"DEXCHUS","title":"USD/CNY","unit":"","lookback":40},
    {"series":"DTWEXBGS","title":"美元指数（广义）","unit":"","lookback":40},
    {"series":"PCOPPUSDM","title":"铜（Proxy）","unit":"$/t","lookback":40},
    {"series":"PALUMUSDM","title":"铝（Proxy）","unit":"$/t","lookback":40},
    {"series":"PZINCUSDM","title":"锌（Proxy）","unit":"$/t","lookback":40},
    {"series":"PNICKUSDM","title":"镍（Proxy）","unit":"$/t","lookback":40},
    {"series":"PLEADUSDM","title":"铅（Proxy）","unit":"$/t","lookback":40},
    {"series":"PTINUSDM","title":"锡（Proxy）","unit":"$/t","lookback":40}
  ],
  "alerts": [
    {"series":"DGS10","op":">","value":4.30,"severity":"warn","message":"融资成本上行：审视现金流与折现敏感业务"},
    {"series":"DEXCHUS","op":">","value":7.20,"severity":"warn","message":"汇率压力：复核进口成本、结算币种与对冲方案"},
    {"series":"PCOPPUSDM","op":"abs>","value":80,"severity":"info","message":"铜价波动放大：复核报价有效期与库存策略"}
  ]
}

# -----------------------------
# CLI / Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="digest_config_v18.json")
    ap.add_argument("--template", required=True, help="daily_digest_template_v18_singlepage.html")
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (BJT)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", default=None, type=int)
    ap.add_argument("--cluster_threshold", default=None, type=float)
    ap.add_argument("--items_per_section", default=None, type=int)
    return ap.parse_args()

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # override from CLI
    if args.limit_raw is not None:
        cfg["limit_raw"] = args.limit_raw
    if args.cluster_threshold is not None:
        cfg["cluster_threshold"] = args.cluster_threshold
    if args.items_per_section is not None:
        cfg["items_per_section"] = args.items_per_section

    date = args.date or date_bjt_str()

    # KPIs + alerts (global, shared across views)
    kpis = build_kpis(cfg)
    alerts = build_alerts(cfg, kpis)

    # Build base sections (cards carry region label)
    base_sections = []
    for s_cfg in cfg.get("sections", []):
        try:
            base_sections.append(build_section(s_cfg, cfg, args))
        except Exception as e:
            base_sections.append({
                "id": s_cfg.get("id") or sha1(s_cfg.get("name",""))[:10],
                "name": s_cfg.get("name",""),
                "cards": [],
                "error": str(e),
            })

    # Build 3 views with their own Top + section briefs (ALL pre-rendered)
    views: dict[str, dict] = {}
    for view in ["all", "cn", "global"]:
        view_sections = split_view_sections(base_sections, view=view)
        # section briefs
        for s in view_sections:
            s["brief"] = section_brief(s["name"], s.get("cards") or [], kpis=kpis, alerts=alerts, model=args.model, target_chars=(300,500))
        top = top_brief(view_name=("全部" if view=="all" else "中国相关" if view=="cn" else "全球（不含中国）"),
                        sections_payload=view_sections, kpis=kpis, alerts=alerts, model=args.model, target_chars=(300,500))
        views[view] = {
            "view": view,
            "top": top,
            "sections": view_sections,
            "nav": nav_counts(view_sections),
        }

    digest = {
        "TITLE": "Ben的每日资讯简报",
        "DATE": date,
        "GENERATED_AT": now_bjt_str(),
        "VERSION": "v18",
        "KPIS": kpis,
        "ALERTS": alerts,
        "VIEWS": views,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render_html(args.template, digest, args.out)

    print("OK:", args.out, "generated_at_bjt=", digest["GENERATED_AT"])
    if not llm_available():
        print("NOTE: 未配置 OPENAI_API_KEY：中文翻译/简报将是弱化版本。建议在 GitHub Secrets 中设置。")

if __name__ == "__main__":
    main()
