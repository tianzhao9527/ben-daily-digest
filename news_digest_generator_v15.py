#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ben Daily Digest Generator (v15)

- Reads config (digest_config_v15.json)
- Fetches global sources (Google News RSS from queries + optional official RSS feeds + arXiv)
- Builds event clusters per section
- Uses OpenAI (optional) to produce:
  - Top brief (300–500 Chinese chars)
  - Section briefs (300–500 Chinese chars each)
  - Event cards with "结论优先" structure, Chinese headline + conclusion + evidence/impact/next
- Renders a single static HTML page (no client-side fetching)

Designed to run reliably on GitHub Actions:
- Few LLM calls (per section, plus 1–3 for top brief)
- Hard timeouts + retries on network calls
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import re
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote_plus

import feedparser
import requests
from jinja2 import Template

# ----------------------------
# Utilities
# ----------------------------

_STOP = set("""
a an the and or for to of in on at by with from as is are was were be been being this that those these
into over under about after before between amid via not no yes
""".split())

def now_beijing() -> _dt.datetime:
    # GitHub Actions runner is UTC; convert to Beijing (+8)
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).astimezone(_dt.timezone(_dt.timedelta(hours=8)))

def parse_dt(entry: dict) -> Optional[_dt.datetime]:
    # feedparser gives struct_time in entry.published_parsed / updated_parsed
    for k in ("published_parsed", "updated_parsed"):
        v = entry.get(k)
        if v:
            try:
                return _dt.datetime.fromtimestamp(time.mktime(v), tz=_dt.timezone.utc)
            except Exception:
                pass
    return None

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    toks = [t for t in s.split() if t and t not in _STOP and len(t) > 1]
    return toks

def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def deny_tld(url: str, deny_tlds: List[str]) -> bool:
    d = domain_of(url)
    for t in deny_tlds or []:
        if d.endswith(t):
            return True
    return False

_CHINA_HINTS = [
    "china","chinese","prc","beijing","shanghai","shenzhen","hong kong","hk","taiwan","macau",
    "cn","cny","rmb","cbirc","pboe","safe","yuan","greater china"
]
def guess_region(text: str) -> str:
    s = (text or "").lower()
    for h in _CHINA_HINTS:
        if h in s:
            return "cn"
    # 中文提示
    if any(x in (text or "") for x in ["中国","大陆","北京","上海","深圳","香港","台湾","澳门","人民币","央行","外汇"]):
        return "cn"
    return "global"

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class RawItem:
    title: str
    url: str
    source: str
    published: Optional[_dt.datetime]
    section_id: str

@dataclass
class Cluster:
    cid: str
    section_id: str
    region: str
    items: List[RawItem]

# ----------------------------
# Sources
# ----------------------------

def google_news_rss(query: str, *, hl="en-US", gl="US", ceid="US:en") -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={hl}&gl={gl}&ceid={ceid}"

def fetch_feed(url: str, *, timeout=(10, 25), max_items: int = 25) -> List[dict]:
    # feedparser can fetch itself, but requests gives us timeouts + headers
    headers = {"User-Agent": "Mozilla/5.0 (digest-bot; +https://github.com)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    return (feed.entries or [])[:max_items]

def fetch_arxiv(arxiv_query: str, *, max_results: int = 40) -> List[dict]:
    q = (arxiv_query or "").strip()
    if not q:
        return []
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query={quote_plus(q)}&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    # arXiv is usually stable; still add timeout
    entries = fetch_feed(url, timeout=(10, 30), max_items=max_results)
    return entries

# ----------------------------
# FRED without API key (CSV endpoint)
# ----------------------------

def fred_csv(series_id: str) -> str:
    # Public CSV endpoint; no API key required
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote_plus(series_id)}"

def fetch_fred_series(series_id: str, lookback: int = 40) -> List[Tuple[str, float]]:
    url = fred_csv(series_id)
    r = requests.get(url, timeout=(10, 25))
    r.raise_for_status()
    lines = r.text.splitlines()
    # header: DATE,VALUE
    pts: List[Tuple[str, float]] = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0].strip(), parts[1].strip()
        if v == "." or v == "":
            continue
        try:
            pts.append((d, float(v)))
        except Exception:
            continue
    if lookback and len(pts) > lookback:
        pts = pts[-lookback:]
    return pts

def spark_svg(values: List[float], w: int = 220, h: int = 44) -> str:
    if not values or len(values) < 2:
        return '<svg class="spark" viewBox="0 0 220 44" xmlns="http://www.w3.org/2000/svg"></svg>'
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    pad = 2
    xs = []
    ys = []
    for i, v in enumerate(values):
        x = pad + (w - 2*pad) * (i / (len(values)-1))
        y = pad + (h - 2*pad) * (1 - (v - vmin) / (vmax - vmin))
        xs.append(x); ys.append(y)
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="2" points="{pts}" />'
        f'</svg>'
    )

# ----------------------------
# OpenAI (optional)
# ----------------------------

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def openai_chat_json(model: str, messages: List[dict], *, timeout=(10, 90), retries: int = 2) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        # JSON mode (supported by many OpenAI chat models). If unsupported, we'll still try to parse.
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err = None
    for i in range(retries + 1):
        try:
            resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            txt = data["choices"][0]["message"]["content"]
            return safe_json_loads(txt)
        except Exception as e:
            last_err = e
            # backoff
            time.sleep(1.5 * (2 ** i))
    raise RuntimeError(f"OpenAI call failed: {last_err}")

def safe_json_loads(text: str) -> dict:
    # tries strict json first; if fails, extract first {...} block
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))

# ----------------------------
# Clustering
# ----------------------------

def cluster_items(section_id: str, items: List[RawItem], thr: float) -> List[Cluster]:
    # Greedy clustering by recency order
    items = sorted(items, key=lambda x: x.published or _dt.datetime(1970,1,1,tzinfo=_dt.timezone.utc), reverse=True)
    clusters: List[Cluster] = []
    for it in items:
        tok = tokenize(it.title)
        best = None
        best_sim = 0.0
        for c in clusters:
            rep = c.items[0]
            sim = jaccard(tok, tokenize(rep.title))
            if sim > best_sim:
                best_sim = sim
                best = c
        if best is not None and best_sim >= thr:
            best.items.append(it)
        else:
            cid = f"{section_id}-{len(clusters)+1}"
            region = guess_region(it.title + " " + it.source + " " + it.url)
            clusters.append(Cluster(cid=cid, section_id=section_id, region=region, items=[it]))
    # within each cluster, sort items by time desc
    for c in clusters:
        c.items.sort(key=lambda x: x.published or _dt.datetime(1970,1,1,tzinfo=_dt.timezone.utc), reverse=True)
    return clusters

# ----------------------------
# Section build + LLM summarization
# ----------------------------

def llm_build_section(model: str, section_name: str, clusters: List[Cluster], items_per_section: int, max_sources: int) -> dict:
    # Prepare compact input
    pack = []
    for c in clusters[:items_per_section]:
        srcs = []
        seen = set()
        for it in c.items:
            d = domain_of(it.url)
            if d in seen:
                continue
            seen.add(d)
            srcs.append({"name": it.source or d or "Source", "title": it.title, "url": it.url})
            if len(srcs) >= min(3, max_sources):
                break
        pack.append({
            "cid": c.cid,
            "region_hint": c.region,
            "titles": [it.title for it in c.items[:3]],
            "sources": srcs
        })

    sys = "你是一个严格、务实的情报分析助手。输出必须是有效 JSON，不要输出解释文字。"
    user = {
        "task": "为一个栏目生成“结论优先”的栏目简报(300-500字中文) + 事件卡列表（10-20条）。",
        "section": section_name,
        "rules": [
            "栏目简报字段：one_liner（<=44字）、facts（3-6条）、impacts（3-6条）、actions（3-6条）、watch（3-6条）、summary（300-500字）",
            "事件卡字段：id=cid, region=('cn'或'global'), headline_zh(中文标题), conclusion(2-3句，先写结论再写原因), evidence(2-4条), impact(2-4条), next(2-4条), tags(3-6个中文标签), sources(1-3条，name+url)",
            "必须中文输出；尽量把标题翻译为自然中文；避免空泛口号，写可执行与可验证的表述。",
            "sources 的 url 必须来自输入 sources；不要编造链接。",
            "如果信息不足，也要给出保守且明确的表述，并在 evidence 中说明“不足/待证”。"
        ],
        "input_clusters": pack
    }

    messages = [{"role":"system","content":sys},{"role":"user","content":json.dumps(user, ensure_ascii=False)}]
    out = openai_chat_json(model, messages)

    # Validate + fill defaults
    brief = out.get("brief") or {}
    cards = out.get("cards") or out.get("events") or []
    if not isinstance(cards, list):
        cards = []
    # normalize
    def _list(x): return x if isinstance(x, list) else []
    brief_norm = {
        "one_liner": normalize_text(brief.get("one_liner","")) or f"{section_name}：今日暂无明确结论",
        "facts": _list(brief.get("facts")),
        "impacts": _list(brief.get("impacts")),
        "actions": _list(brief.get("actions")),
        "watch": _list(brief.get("watch")),
        "summary": normalize_text(brief.get("summary","")) or "今日暂无足够可用信息形成可靠简报（可能是源抓取失败或事件稀缺）。"
    }

    cards_norm = []
    for c in cards[:items_per_section]:
        cid = c.get("id") or c.get("cid")
        if not cid:
            continue
        region = c.get("region") or "global"
        if region not in ("cn","global"):
            region = "cn" if guess_region(c.get("headline_zh","")+ " " + json.dumps(c.get("sources",[]), ensure_ascii=False))=="cn" else "global"
        srcs = c.get("sources") or []
        if not isinstance(srcs, list):
            srcs = []
        srcs2 = []
        for s in srcs[:max_sources]:
            if isinstance(s, dict) and s.get("url"):
                srcs2.append({"name": s.get("name") or domain_of(s["url"]) or "Source", "url": s["url"]})
        cards_norm.append({
            "id": str(cid),
            "region": region,
            "headline_zh": normalize_text(c.get("headline_zh","")) or normalize_text(c.get("headline","")) or "（无标题）",
            "conclusion": normalize_text(c.get("conclusion","")) or "结论：信息不足，需继续观察。",
            "evidence": _list(c.get("evidence")),
            "impact": _list(c.get("impact")),
            "next": _list(c.get("next")),
            "tags": _list(c.get("tags")),
            "sources": srcs2
        })

    return {"brief": brief_norm, "cards": cards_norm}

def llm_top_brief(model: str, sections: List[dict], kpis: List[dict], view_name: str) -> dict:
    # Create compact prompt
    sec_pack = []
    for s in sections:
        b = s.get("brief") or {}
        sec_pack.append({
            "name": s.get("name"),
            "one_liner": b.get("one_liner"),
            "facts": (b.get("facts") or [])[:4],
            "watch": (b.get("watch") or [])[:4],
        })
    kpi_pack = []
    for k in kpis[:9]:
        kpi_pack.append({
            "title": k.get("title"),
            "id": k.get("id"),
            "value": k.get("value"),
            "delta_pct": k.get("delta_pct"),
            "direction": k.get("direction"),
            "unit": k.get("unit"),
        })

    sys = "你是一个务实的全球宏观与产业情报编辑。输出必须是有效 JSON，不要输出解释文字。"
    user = {
        "task": f"为“{view_name}”生成 Top 今日要点（结构化 + 300-500字中文总结）。",
        "rules": [
            "字段：one_liner（<=44字）、facts（3-6条）、actions（3-6条）、watch（3-6条）、summary（300-500字）",
            "写实，避免空话；facts 必须尽量引用 KPI/栏目事实锚点；watch 写清楚触发阈值/方向（如果无法给阈值，就写“若继续上行/下行需关注”）。"
        ],
        "sections": sec_pack,
        "kpis": kpi_pack
    }
    messages = [{"role":"system","content":sys},{"role":"user","content":json.dumps(user, ensure_ascii=False)}]
    out = openai_chat_json(model, messages)
    top = out.get("top") or out
    def _list(x): return x if isinstance(x, list) else []
    return {
        "one_liner": normalize_text(top.get("one_liner","")) or f"{view_name}：今日关注风险偏好与关键政策信号",
        "facts": _list(top.get("facts")),
        "actions": _list(top.get("actions")),
        "watch": _list(top.get("watch")),
        "summary": normalize_text(top.get("summary","")) or "今日暂无足够信息形成可靠要点（可能是源抓取失败或事件稀缺）。"
    }

# ----------------------------
# Rendering
# ----------------------------

def render_html(template_path: str, context: dict) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())
    return template.render(**context)

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (Beijing). default=today")
    ap.add_argument("--model", default=os.getenv("DIGEST_MODEL", "gpt-4o-mini"))
    ap.add_argument("--limit_raw", type=int, default=None)
    ap.add_argument("--cluster_threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    items_per_section = int(cfg.get("items_per_section", 15))
    thr = float(args.cluster_threshold if args.cluster_threshold is not None else cfg.get("cluster_threshold", 0.42))
    limit_raw = int(args.limit_raw if args.limit_raw is not None else cfg.get("limit_raw", 60))
    max_sources = int(cfg.get("max_sources", 5))
    deny_tlds = cfg.get("deny_tlds", [])

    bj_now = now_beijing()
    if args.date:
        date_str = args.date
    else:
        date_str = bj_now.strftime("%Y-%m-%d")

    print("[digest] boot", flush=True)
    print(f"[digest] date={date_str} model={args.model} limit_raw={limit_raw} thr={thr} items={items_per_section}", flush=True)

    # 1) KPI
    kpis_cfg = cfg.get("kpis", [])
    kpis: List[dict] = []
    print(f"[digest] kpi start: {len(kpis_cfg)}", flush=True)
    for k in kpis_cfg:
        sid = k.get("series")
        title = k.get("title", sid)
        unit = k.get("unit","")
        lookback = int(k.get("lookback", 40))
        try:
            pts = fetch_fred_series(sid, lookback=lookback)
            vals = [v for _, v in pts]
            value = vals[-1] if vals else None
            delta_abs = (vals[-1] - vals[-2]) if len(vals) >= 2 else None
            delta_pct = ((delta_abs / vals[-2]) * 100.0) if (delta_abs is not None and vals[-2] != 0) else None
            direction = "flat"
            if delta_abs is not None:
                if delta_abs > 0: direction = "up"
                elif delta_abs < 0: direction = "down"
            kpis.append({
                "id": sid,
                "title": title,
                "unit": unit,
                "value": value,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "direction": direction,
                "source": "FRED",
                "spark": spark_svg(vals)
            })
        except Exception as e:
            kpis.append({
                "id": sid,
                "title": title,
                "unit": unit,
                "value": None,
                "delta_abs": None,
                "delta_pct": None,
                "direction": "flat",
                "source": "FRED",
                "spark": spark_svg([])
            })
    print(f"[digest] kpi done: {len(kpis)}", flush=True)

    # 2) Alerts
    alerts_cfg = cfg.get("alerts", [])
    alerts: List[dict] = []
    kpi_map = {k["id"]: k for k in kpis}
    for a in alerts_cfg:
        sid = a.get("series")
        op = a.get("op")
        thr_v = a.get("value")
        sev = a.get("severity","info")
        msg = a.get("message","")
        k = kpi_map.get(sid)
        if not k or k.get("value") is None:
            continue
        v = float(k["value"])
        hit = False
        if op == ">" and v > float(thr_v): hit = True
        elif op == "<" and v < float(thr_v): hit = True
        elif op == "abs>" and abs(v) > float(thr_v): hit = True
        if hit:
            alerts.append({
                "severity": sev,
                "title": k.get("title", sid),
                "op": op,
                "threshold": thr_v,
                "message": msg
            })

    # 3) Sections: fetch raw
    sections_cfg = cfg.get("sections", [])
    print(f"[digest] sections start: {len(sections_cfg)}", flush=True)

    all_sections: List[dict] = []
    raw_items_all: List[RawItem] = []

    for s in sections_cfg:
        sid = s.get("id")
        sname = s.get("name", sid)
        stype = s.get("type","")
        print(f"[digest] section start: {sid} ({sname})", flush=True)

        items: List[RawItem] = []

        # feeds (official)
        for f in s.get("feeds", []) or []:
            try:
                url = f.get("url")
                src = f.get("source") or "Feed"
                if not url:
                    continue
                entries = fetch_feed(url, max_items=25)
                for e in entries:
                    title = normalize_text(e.get("title",""))
                    link = e.get("link") or e.get("id") or ""
                    if not title or not link:
                        continue
                    if deny_tld(link, deny_tlds):
                        continue
                    dt = parse_dt(e)
                    items.append(RawItem(title=title, url=link, source=src, published=dt, section_id=sid))
                print(f"[digest] feed {sid}:{src} entries={len(entries)}", flush=True)
            except Exception as ex:
                print(f"[digest] feed {sid} failed: {ex}", flush=True)

        # queries -> Google News RSS
        for q in s.get("queries", []) or []:
            try:
                rss = google_news_rss(q)
                if not rss:
                    continue
                entries = fetch_feed(rss, max_items=25)
                for e in entries:
                    title = normalize_text(e.get("title",""))
                    link = e.get("link") or ""
                    if not title or not link:
                        continue
                    if deny_tld(link, deny_tlds):
                        continue
                    # Google News titles often: "headline - Source"
                    src = "Google News"
                    dt = parse_dt(e)
                    items.append(RawItem(title=title, url=link, source=src, published=dt, section_id=sid))
                print(f"[digest] gnews {sid} q='{q[:30]}...' entries={len(entries)}", flush=True)
            except Exception as ex:
                print(f"[digest] gnews {sid} failed: {ex}", flush=True)

        # arxiv
        if (s.get("type") == "arxiv") and s.get("arxiv_query"):
            try:
                entries = fetch_arxiv(s["arxiv_query"], max_results=40)
                for e in entries:
                    title = normalize_text(e.get("title",""))
                    link = e.get("link") or ""
                    if not title or not link:
                        continue
                    if deny_tld(link, deny_tlds):
                        continue
                    dt = parse_dt(e)
                    items.append(RawItem(title=title, url=link, source="arXiv", published=dt, section_id=sid))
                print(f"[digest] arxiv {sid} entries={len(entries)}", flush=True)
            except Exception as ex:
                print(f"[digest] arxiv {sid} failed: {ex}", flush=True)

        # Deduplicate by URL then title
        seen_url = set()
        uniq: List[RawItem] = []
        for it in sorted(items, key=lambda x: x.published or _dt.datetime(1970,1,1,tzinfo=_dt.timezone.utc), reverse=True):
            if it.url in seen_url:
                continue
            seen_url.add(it.url)
            uniq.append(it)
        # cap raw
        uniq = uniq[:limit_raw]
        raw_items_all.extend(uniq)

        clusters = cluster_items(sid, uniq, thr)
        # cap clusters
        clusters = clusters[:items_per_section]

        section_out = {
            "id": sid,
            "name": sname,
            "clusters": clusters,  # keep for debug
            "brief": {
                "one_liner": f"{sname}：待生成",
                "facts": [],
                "impacts": [],
                "actions": [],
                "watch": [],
                "summary": ""
            },
            "cards": []
        }

        # LLM section build
        api_key = os.getenv("OPENAI_API_KEY","").strip()
        if api_key and clusters:
            try:
                built = llm_build_section(args.model, sname, clusters, items_per_section, max_sources)
                section_out["brief"] = built["brief"]
                section_out["cards"] = built["cards"]
            except Exception as ex:
                print(f"[digest] llm section {sid} failed: {ex}", flush=True)
                # fallback: make simple cards from cluster reps
                section_out["brief"]["one_liner"] = f"{sname}：源可用但摘要失败"
                section_out["brief"]["summary"] = "自动摘要失败（LLM调用异常/超时）。建议稍后重试。"
                for c in clusters[:items_per_section]:
                    rep = c.items[0]
                    section_out["cards"].append({
                        "id": c.cid,
                        "region": c.region,
                        "headline_zh": rep.title,
                        "conclusion": "结论：已出现相关报道，但需要进一步核验细节。",
                        "evidence": [rep.title],
                        "impact": ["待评估"],
                        "next": ["继续跟踪更多权威来源"],
                        "tags": ["跟踪"],
                        "sources": [{"name": rep.source, "url": rep.url}]
                    })
        else:
            # no key or empty
            section_out["brief"]["one_liner"] = f"{sname}：今日未抓取到足够条目"
            section_out["brief"]["summary"] = "本栏目今日未抓取到足够条目（可能是源访问失败/查询过窄）。"
            section_out["cards"] = []

        all_sections.append(section_out)
        print(f"[digest] section done: {sid} events={len(section_out['cards'])}", flush=True)

    print("[digest] sections done", flush=True)

    # 4) Build views (all/cn/global)
    def filter_sections(region: str) -> List[dict]:
        out = []
        for s in all_sections:
            cards = s.get("cards") or []
            if region == "all":
                cc = cards
            elif region == "cn":
                cc = [c for c in cards if c.get("region") == "cn"]
            else:
                cc = [c for c in cards if c.get("region") != "cn"]
            # keep section even if empty: template expects sections list; but nav count will be 0
            out.append({
                "id": s["id"],
                "name": s["name"],
                "brief": s.get("brief") or {},
                "cards": cc
            })
        return out

    sections_all = filter_sections("all")
    sections_cn = filter_sections("cn")
    sections_global = filter_sections("global")

    # nav lists
    def make_nav(sections: List[dict]) -> List[dict]:
        return [{"id": s["id"], "name": s["name"], "count": len(s.get("cards") or [])} for s in sections]

    # 5) Top brief (LLM) — one per view, but fall back if LLM unavailable
    api_key = os.getenv("OPENAI_API_KEY","").strip()
    if api_key:
        try:
            top_all = llm_top_brief(args.model, sections_all, kpis, "全部")
        except Exception as ex:
            print(f"[digest] llm top all failed: {ex}", flush=True)
            top_all = {"one_liner":"今日要点：摘要生成失败","facts":[],"actions":[],"watch":[],"summary":"Top 摘要生成失败（LLM调用异常/超时）。"}
        try:
            top_cn = llm_top_brief(args.model, sections_cn, kpis, "中国相关")
        except Exception:
            top_cn = top_all
        try:
            top_global = llm_top_brief(args.model, sections_global, kpis, "全球（不含中国）")
        except Exception:
            top_global = top_all
    else:
        top_all = {"one_liner":"今日要点：未配置 OPENAI_API_KEY","facts":[],"actions":[],"watch":[],"summary":"未配置 OPENAI_API_KEY，无法生成中文结构化要点。"}
        top_cn = top_all
        top_global = top_all

    # Ensure required keys exist for template
    def ensure_top(top: dict) -> dict:
        return {
            "one_liner": normalize_text(top.get("one_liner","")) or "今日要点：暂无",
            "facts": top.get("facts") if isinstance(top.get("facts"), list) else [],
            "actions": top.get("actions") if isinstance(top.get("actions"), list) else [],
            "watch": top.get("watch") if isinstance(top.get("watch"), list) else [],
            "summary": normalize_text(top.get("summary","")) or ""
        }

    # Ensure section brief keys
    def ensure_brief(b: dict, name: str) -> dict:
        if not isinstance(b, dict):
            b = {}
        return {
            "one_liner": normalize_text(b.get("one_liner","")) or f"{name}：暂无",
            "facts": b.get("facts") if isinstance(b.get("facts"), list) else [],
            "impacts": b.get("impacts") if isinstance(b.get("impacts"), list) else [],
            "actions": b.get("actions") if isinstance(b.get("actions"), list) else [],
            "watch": b.get("watch") if isinstance(b.get("watch"), list) else [],
            "summary": normalize_text(b.get("summary","")) or ""
        }

    for s in sections_all:
        s["brief"] = ensure_brief(s.get("brief"), s["name"])
    for s in sections_cn:
        s["brief"] = ensure_brief(s.get("brief"), s["name"])
    for s in sections_global:
        s["brief"] = ensure_brief(s.get("brief"), s["name"])

    views = {
        "all": {"sections": sections_all, "nav": make_nav(sections_all), "top": ensure_top(top_all)},
        "cn": {"sections": sections_cn, "nav": make_nav(sections_cn), "top": ensure_top(top_cn)},
        "global": {"sections": sections_global, "nav": make_nav(sections_global), "top": ensure_top(top_global)},
    }

    # 6) Render
    print("[digest] render html", flush=True)
    context = {
        "TITLE": "Ben的每日资讯简报",
        "DATE": date_str,
        "GENERATED_AT": bj_now.strftime("%H:%M"),
        "views": views,
        "VIEWS": views,   # template JS expects VIEWS
        "KPIS": kpis,
        "ALERTS": alerts
    }
    html = render_html(args.template, context)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[digest] wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
