#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py
Generate a single static HTML digest page by:
- fetching Google News RSS (and optional RSS feeds / arXiv),
- clustering into "event cards",
- producing Chinese titles/summaries + 300–500字 section briefs via OpenAI (batched per section),
- embedding everything into a static HTML template (no client-side fetching required).

Designed to be stable on GitHub Actions:
- hard timeouts for HTTP + OpenAI
- retry with backoff
- SIGUSR1 stack dump support for watchdog diagnostics
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
from urllib.parse import quote_plus, urlparse, parse_qs

import faulthandler
import signal

import requests
import feedparser

try:
    from jinja2 import Template  # optional; template might not be Jinja-heavy
except Exception:
    Template = None  # type: ignore


# ----------------------------
# Diagnostics: stack dump
# ----------------------------
faulthandler.enable(all_threads=True)
try:
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
except Exception:
    pass


# ----------------------------
# HTTP helpers
# ----------------------------
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"

def http_get(url: str, *, timeout: Tuple[int, int] = (10, 25), headers: Optional[Dict[str, str]] = None) -> bytes:
    h = {"User-Agent": UA, "Accept": "*/*"}
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    return r.content

def http_get_text(url: str, *, timeout: Tuple[int, int] = (10, 25), headers: Optional[Dict[str, str]] = None) -> str:
    return http_get(url, timeout=timeout, headers=headers).decode("utf-8", errors="ignore")

def retry(fn, *, tries: int = 3, base: float = 1.2, jitter: float = 0.25, what: str = ""):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep = base ** i + random.random() * jitter
            if what:
                print(f"[digest] retry {what}: {type(e).__name__}: {e} (sleep {sleep:.1f}s)", flush=True)
            time.sleep(sleep)
    raise last  # type: ignore


# ----------------------------
# Date helpers
# ----------------------------
def now_bjt() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))

def iso_date(d: dt.date) -> str:
    return d.isoformat()

def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


# ----------------------------
# URL unwrap (best-effort)
# ----------------------------
def unwrap_google_news_url(link: str) -> str:
    """
    Google News RSS links are often redirect wrappers. Try to extract the real url if present.
    """
    try:
        u = urlparse(link)
        qs = parse_qs(u.query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]
    except Exception:
        pass
    return link


# ----------------------------
# Google News RSS
# ----------------------------
def google_news_rss(query: str, *, hl="en-US", gl="US", ceid="US:en") -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"


# ----------------------------
# Tokenization / similarity
# ----------------------------
STOP = set("""
a an the and or but if then else for to of in on at by from with without into over under
is are was were be been being as that this these those it its their his her our your
update updates latest today says said report reports amid after before
""".split())

def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def token_set(title: str) -> set:
    t = norm_text(title).lower()
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t)
    parts = [p for p in t.split() if p and p not in STOP and len(p) > 1]
    return set(parts)

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)

def map_threshold(thr: float) -> float:
    """
    Config uses embedding-like thresholds (0.7~0.9). For Jaccard, map to practical values.
    """
    thr = float(thr)
    if thr >= 0.85:
        return 0.33
    if thr >= 0.80:
        return 0.30
    if thr >= 0.75:
        return 0.27
    return 0.25


# ----------------------------
# Region classifier (heuristic)
# ----------------------------
CN_HINT = re.compile(
    r"(\bchina\b|\bchinese\b|beijing|shanghai|shenzhen|hong kong|taiwan|macau|"
    r"\bcny\b|\brmb\b|pboc|csrc|ndrc|mofcom|safe|cnhi|onshore|offshore|"
    r"\bhk\b|\bcn\b|tencent|alibaba|byd|huawei|smic|evergrande|"
    r"中国|大陆|北京|上海|深圳|香港|台湾|澳门|人民币|央行|发改委|商务部|证监会)"
    , re.I
)

def classify_region(item: Dict[str, Any]) -> str:
    title = safe_str(item.get("title",""))
    src = safe_str(item.get("source",""))
    url = safe_str(item.get("url",""))
    if CN_HINT.search(title) or CN_HINT.search(src):
        return "cn"
    # Chinese characters presence -> likely CN-related
    if re.search(r"[\u4e00-\u9fff]", title):
        return "cn"
    # tld
    try:
        host = urlparse(url).netloc.lower()
        if host.endswith(".cn"):
            return "cn"
    except Exception:
        pass
    return "global"


# ----------------------------
# Data classes
# ----------------------------
@dataclasses.dataclass
class RawItem:
    title: str
    url: str
    date: str
    source: str
    snippet: str
    region: str

@dataclasses.dataclass
class EventCard:
    id: str
    title: str
    title_zh: str
    summary_zh: str
    url: str
    date: str
    source_hint: str
    region: str
    tags: List[str]

@dataclasses.dataclass
class SectionOut:
    id: str
    name: str
    brief_zh: str
    brief_cn: str
    brief_global: str
    tags: List[str]
    events: List[Dict[str, Any]]


# ----------------------------
# RSS / arXiv fetch
# ----------------------------
def parse_rss(content: bytes) -> feedparser.FeedParserDict:
    return feedparser.parse(content)

def fetch_google_news_items(section_id: str, query: str, limit: int, *, timeout: Tuple[int,int]) -> List[Dict[str, Any]]:
    url = google_news_rss(query)
    def _do():
        return http_get(url, timeout=timeout)
    content = retry(_do, tries=3, what=f"gnews {section_id}")
    feed = parse_rss(content)
    items = []
    for e in feed.entries[:limit]:
        title = norm_text(getattr(e, "title", "") or "")
        link = getattr(e, "link", "") or ""
        # Google often provides source
        src = ""
        try:
            if hasattr(e, "source") and e.source:
                src = safe_str(getattr(e.source, "title", "")) or safe_str(getattr(e.source, "href", ""))
        except Exception:
            src = ""
        if not src:
            # fallback: feed title
            src = safe_str(getattr(feed.feed, "title", "")) or "Google News"
        published = safe_str(getattr(e, "published", "")) or safe_str(getattr(e, "updated", ""))
        snippet = ""
        snippet = safe_str(getattr(e, "summary", "")) or safe_str(getattr(e, "description", ""))
        snippet = re.sub(r"<[^>]+>", "", snippet)
        items.append({
            "title": title,
            "url": unwrap_google_news_url(link),
            "date": published[:25],
            "source": src,
            "snippet": norm_text(snippet)[:320],
        })
    print(f"[digest] gnews {section_id} q='{query[:28]}...' entries={len(feed.entries)}", flush=True)
    return items

def fetch_feed_items(section_id: str, feed_url: str, limit: int, *, timeout: Tuple[int,int]) -> List[Dict[str, Any]]:
    def _do():
        return http_get(feed_url, timeout=timeout)
    try:
        content = retry(_do, tries=2, what=f"feed {section_id}")
    except Exception as e:
        print(f"[digest] feed {section_id} failed: {e}", flush=True)
        return []
    feed = parse_rss(content)
    out = []
    for e in feed.entries[:limit]:
        title = norm_text(getattr(e, "title", "") or "")
        link = getattr(e, "link", "") or ""
        src = safe_str(getattr(feed.feed, "title", "")) or urlparse(feed_url).netloc
        published = safe_str(getattr(e, "published", "")) or safe_str(getattr(e, "updated", ""))
        snippet = safe_str(getattr(e, "summary", "")) or safe_str(getattr(e, "description", ""))
        snippet = re.sub(r"<[^>]+>", "", snippet)
        out.append({
            "title": title,
            "url": link,
            "date": published[:25],
            "source": src,
            "snippet": norm_text(snippet)[:320],
        })
    return out

def fetch_arxiv(section_id: str, query: str, limit: int, *, timeout: Tuple[int,int]) -> List[Dict[str, Any]]:
    # arXiv API: Atom
    q = quote_plus(query)
    url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={min(limit,50)}&sortBy=submittedDate&sortOrder=descending"
    try:
        content = retry(lambda: http_get(url, timeout=timeout), tries=2, what=f"arxiv {section_id}")
    except Exception as e:
        print(f"[digest] arxiv fetch failed: {e}", flush=True)
        return []
    feed = feedparser.parse(content)
    out=[]
    for e in feed.entries[:limit]:
        title = norm_text(getattr(e, "title","") or "")
        link = ""
        for l in getattr(e,"links",[]) or []:
            if safe_str(l.get("rel",""))=="alternate":
                link = safe_str(l.get("href",""))
                break
        if not link:
            link = safe_str(getattr(e,"link",""))
        published = safe_str(getattr(e,"published",""))[:25]
        snippet = norm_text(re.sub(r"\s+"," ", safe_str(getattr(e,"summary",""))))[:360]
        out.append({
            "title": title,
            "url": link,
            "date": published,
            "source": "arXiv",
            "snippet": snippet
        })
    return out


# ----------------------------
# De-dup and rank
# ----------------------------
def canonical_key(item: Dict[str, Any]) -> str:
    url = safe_str(item.get("url","")).strip()
    title = safe_str(item.get("title","")).strip().lower()
    # normalize google rss wrapper urls
    if "news.google.com" in url and "url=" in url:
        url = unwrap_google_news_url(url)
    return hashlib.sha1((url+"|"+title).encode("utf-8", errors="ignore")).hexdigest()

def dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen=set()
    out=[]
    for it in items:
        k=canonical_key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

def parse_date(s: str) -> float:
    # best-effort: many RSS dates are not ISO
    s = (s or "").strip()
    if not s:
        return 0.0
    # try feedparser parsing
    try:
        tt = feedparser._parse_date(s)  # type: ignore
        if tt:
            return time.mktime(tt)
    except Exception:
        pass
    # try ISO
    try:
        return dt.datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()
    except Exception:
        return 0.0

def item_score(it: Dict[str, Any]) -> float:
    # recency + snippet availability
    ts = parse_date(safe_str(it.get("date","")))
    rec = 0.0
    if ts>0:
        rec = (ts / 1e9)  # coarse
    sn = 0.15 if safe_str(it.get("snippet","")) else 0.0
    return rec + sn


# ----------------------------
# Clustering -> clusters of items
# ----------------------------
def cluster_items(items: List[Dict[str, Any]], thr_embed: float) -> List[List[Dict[str, Any]]]:
    thr = map_threshold(thr_embed)
    clusters: List[Tuple[set, List[Dict[str, Any]]]] = []
    for it in items:
        tset = token_set(safe_str(it.get("title","")))
        placed = False
        for idx,(cset, lst) in enumerate(clusters):
            sim = jaccard(tset, cset)
            if sim >= thr:
                lst.append(it)
                # update centroid token set (union)
                clusters[idx] = (cset | tset, lst)
                placed = True
                break
        if not placed:
            clusters.append((tset, [it]))
    # return only lists, sorted by size+score
    def c_score(lst):
        return (len(lst), max(item_score(x) for x in lst))
    out = [lst for _,lst in clusters]
    out.sort(key=c_score, reverse=True)
    return out


# ----------------------------
# LLM (OpenAI) batched JSON
# ----------------------------
def openai_chat_json(model: str, messages: List[Dict[str, str]], *, timeout: Tuple[int,int]=(10,60), max_retries: int=3) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY","").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": messages,
        "response_format": {"type":"json_object"},
    }

    def _do():
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    data = retry(_do, tries=max_retries, what="openai")
    try:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"OpenAI JSON parse failed: {e}")

def llm_section_pack(model: str, section_name: str, events: List[EventCard], *, want_brief_chars: Tuple[int,int]=(300,500)) -> Tuple[str, List[Dict[str, str]]]:
    """
    Return (section_brief_zh, list of {id,title_zh,summary_zh,tags})
    Single call per section.
    """
    # limit payload size
    pack = []
    for ev in events:
        pack.append({
            "id": ev.id,
            "title": ev.title,
            "snippet": ev.summary_zh or "",  # we may put english snippet in summary_zh temporarily
            "source": ev.source_hint,
            "date": ev.date
        })

    sys_prompt = (
        "你是专业的全球资讯编辑。输出必须是严格 JSON（不要代码块）。"
        "要求：中文、具体、结论优先，避免空泛。每条事件卡总结 2-3 句，包含关键主体/数字/动作（如果有）。"
        f"并为本板块写一段 {want_brief_chars[0]}–{want_brief_chars[1]} 字的板块简报（中文），突出1-3个最重要结论。"
    )

    user_prompt = {
        "section": section_name,
        "events": pack,
        "schema": {
            "brief_zh": "string",
            "events": [{"id":"string","title_zh":"string","summary_zh":"string","tags":["string"]}]
        }
    }

    messages = [
        {"role":"system","content": sys_prompt},
        {"role":"user","content": json.dumps(user_prompt, ensure_ascii=False)}
    ]
    out = openai_chat_json(model, messages)
    brief = safe_str(out.get("brief_zh","")).strip()
    evs = out.get("events", [])
    if not isinstance(evs, list):
        evs = []
    # normalize
    mapped=[]
    for x in evs:
        if not isinstance(x, dict):
            continue
        mapped.append({
            "id": safe_str(x.get("id","")),
            "title_zh": safe_str(x.get("title_zh","")).strip(),
            "summary_zh": safe_str(x.get("summary_zh","")).strip(),
            "tags": x.get("tags", []) if isinstance(x.get("tags", []), list) else []
        })
    return brief, mapped

def llm_daily_brief(model: str, top_events: List[EventCard], kpis: Dict[str, Any]) -> str:
    pack=[]
    for ev in top_events[:12]:
        pack.append({"title": ev.title, "source": ev.source_hint, "date": ev.date})
    kpi_lines=[]
    for group in ("macro","metals"):
        for k in (kpis.get(group) or [])[:4]:
            try:
                kpi_lines.append(f"{k.get('label')}: {k.get('value_str')}（Δ {k.get('delta','—')}）")
            except Exception:
                continue

    sys_prompt = (
        "你是专业的全球宏观与产业研究助理。请写一段 300–500 字的《今日要点》中文简报，"
        "结构清晰（3-5条要点，包含因果/风险/机会），避免空话。"
    )
    user_prompt = {"top_events": pack, "kpis": kpi_lines}
    messages = [{"role":"system","content":sys_prompt},{"role":"user","content":json.dumps(user_prompt,ensure_ascii=False)}]
    out = openai_chat_json(model, messages)
    # allow either {"brief":"..."} or {"today":"..."}
    for key in ("brief","today","summary","top"):
        if key in out:
            return safe_str(out.get(key,"")).strip()
    # fallback: take first string value
    for v in out.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ----------------------------
# KPI from FRED
# ----------------------------
def fred_csv_url(series: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote_plus(series)}"

def spark_svg(values: List[float], *, stroke: str="rgba(128,128,128,.6)") -> str:
    vals=[v for v in values if v is not None and not (isinstance(v,float) and math.isnan(v))]
    if len(vals) < 2:
        return ""
    lo=min(vals); hi=max(vals)
    rng = hi-lo if hi!=lo else 1.0
    pts=[]
    for i,v in enumerate(vals[-40:]):
        x = i/(len(vals[-40:])-1)*100
        y = 42 - ((v-lo)/rng)*36
        pts.append(f"{x:.1f},{y:.1f}")
    return f'<svg class="spark" viewBox="0 0 100 44"><polyline points="{" ".join(pts)}" stroke="{stroke}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'

def fetch_kpis(cfg: Dict[str, Any], *, timeout: Tuple[int,int]) -> Dict[str, Any]:
    kcfg = cfg.get("kpis", []) or []
    out_macro=[]
    out_metals=[]
    for k in kcfg:
        series = safe_str(k.get("series",""))
        label = safe_str(k.get("title","")) or series
        unit = safe_str(k.get("unit",""))
        lookback = int(k.get("lookback", 40))
        url = fred_csv_url(series)
        def _do():
            return http_get_text(url, timeout=timeout)
        try:
            txt = retry(_do, tries=2, what=f"fred {series}")
        except Exception as e:
            print(f"[digest] kpi fail {series}: {e}", flush=True)
            continue
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        vals=[]
        for ln in lines[1:]:
            parts=ln.split(",")
            if len(parts)<2: 
                continue
            v=parts[1].strip()
            if v=="." or v=="":
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if len(vals) < 2:
            continue
        latest=vals[-1]; prev=vals[-2]
        delta = latest - prev
        pct = None
        try:
            pct = round(delta/prev*100, 2) if prev!=0 else None
        except Exception:
            pct = None
        # colors: green up, red down
        stroke = "#16a34a" if delta>0 else ("#dc2626" if delta<0 else "rgba(128,128,128,.6)")
        obj = {
            "label": label,
            "unit": unit,
            "value": latest,
            "value_str": f"{latest:.3f}" if abs(latest) < 1000 else f"{latest:,.2f}",
            "delta": round(delta, 3),
            "pct": pct,
            "source": "FRED",
            "spark_svg": spark_svg(vals[-lookback:], stroke=stroke),
        }
        # classify
        if "铜" in label or "铝" in label or "锌" in label or "镍" in label or "铅" in label or "锡" in label or series.startswith("P"):
            out_metals.append(obj)
        else:
            out_macro.append(obj)
    return {
        "generated_at_bjt": now_bjt().strftime("%Y-%m-%d %H:%M"),
        "macro": out_macro,
        "metals": out_metals
    }


# ----------------------------
# Build digest
# ----------------------------
def event_id_from(title: str, url: str) -> str:
    base = (url or "") + "|" + (title or "")
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:14]

def build_events_for_section(sec_cfg: Dict[str, Any], cfg: Dict[str, Any], *, timeout_http: Tuple[int,int], model: str) -> Tuple[List[EventCard], str]:
    sec_id = safe_str(sec_cfg.get("id",""))
    sec_name = safe_str(sec_cfg.get("name","")) or sec_id
    items_per_section = int(cfg.get("items_per_section", 15))
    thr = float(cfg.get("cluster_threshold", 0.82))
    limit_raw = int(cfg.get("limit_raw", 25))

    print(f"[digest] section start: {sec_id} ({sec_name})", flush=True)

    raw: List[Dict[str, Any]] = []
    # feeds (optional)
    for fu in sec_cfg.get("feeds", []) or []:
        raw.extend(fetch_feed_items(sec_id, fu, limit_raw, timeout=timeout_http))

    # arxiv section
    if safe_str(sec_cfg.get("type","")) == "arxiv":
        q = safe_str(sec_cfg.get("arxiv_query","")) or "cat:cs.AI"
        raw.extend(fetch_arxiv(sec_id, q, limit_raw, timeout=timeout_http))
    else:
        for q in sec_cfg.get("queries", []) or []:
            raw.extend(fetch_google_news_items(sec_id, safe_str(q), limit_raw, timeout=timeout_http))

    raw = dedup(raw)
    # rank by score
    raw.sort(key=item_score, reverse=True)
    raw = raw[: max(items_per_section*3, 60)]

    # normalize to RawItem list and region tags
    normed=[]
    for it in raw:
        it2 = dict(it)
        it2["region"] = classify_region(it2)
        normed.append(it2)

    clusters = cluster_items(normed, thr)
    # convert clusters -> preliminary events (english titles still)
    prelim: List[EventCard] = []
    for cl in clusters:
        # pick representative by score
        cl_sorted = sorted(cl, key=item_score, reverse=True)
        rep = cl_sorted[0]
        title = safe_str(rep.get("title",""))
        url = safe_str(rep.get("url",""))
        date = safe_str(rep.get("date",""))
        region = safe_str(rep.get("region","global"))
        source = safe_str(rep.get("source",""))
        n = len(cl_sorted)
        source_hint = source if n<=1 else f"{source} · 另{n-1}篇"
        snippet = safe_str(rep.get("snippet",""))
        ev = EventCard(
            id=event_id_from(title, url),
            title=title,
            title_zh="",
            summary_zh=snippet,  # temporarily store english snippet
            url=url,
            date=date,
            source_hint=source_hint,
            region=region,
            tags=[]
        )
        prelim.append(ev)
        if len(prelim) >= items_per_section:
            break

    # fallback: if clustering yields none, take top N raw items
    if not prelim:
        for it in normed[:items_per_section]:
            title = safe_str(it.get("title",""))
            url = safe_str(it.get("url",""))
            ev = EventCard(
                id=event_id_from(title, url),
                title=title,
                title_zh="",
                summary_zh=safe_str(it.get("snippet","")),
                url=url,
                date=safe_str(it.get("date","")),
                source_hint=safe_str(it.get("source","")),
                region=safe_str(it.get("region","global")),
                tags=[]
            )
            prelim.append(ev)

    # Single LLM call: section brief + chinese titles/summaries
    brief_zh = ""
    mapped = []
    try:
        brief_zh, mapped = llm_section_pack(model, sec_name, prelim)
    except Exception as e:
        print(f"[digest] llm section failed {sec_id}: {e}", flush=True)

    # apply mapped outputs
    by_id = {x["id"]: x for x in mapped if x.get("id")}
    final=[]
    for ev in prelim:
        m = by_id.get(ev.id, {})
        ev.title_zh = safe_str(m.get("title_zh","")).strip() or ev.title
        ev.summary_zh = safe_str(m.get("summary_zh","")).strip() or ev.summary_zh
        tags = m.get("tags", [])
        ev.tags = [safe_str(t) for t in tags] if isinstance(tags, list) else []
        final.append(ev)

    print(f"[digest] section done: {sec_id} events={len(final)}", flush=True)
    if not brief_zh:
        brief_zh = "本板块今日暂无足够可用内容（可能因信息源受限或抓取为空）。"
    return final, brief_zh

def make_top_section(all_sections: List[SectionOut], model: str, kpis: Dict[str, Any], *, items: int=12) -> SectionOut:
    # collect top events across sections
    pool=[]
    for sec in all_sections:
        for ev in sec.events:
            # ev is dict
            pool.append(ev)
    # simple rank: by date parse + presence
    def s(ev):
        return parse_date(ev.get("date","")) + (0.5 if ev.get("source_hint","") else 0.0)
    pool.sort(key=s, reverse=True)
    top_events = pool[:items]
    # build EventCard list for LLM brief
    top_cards=[]
    for ev in top_events:
        top_cards.append(EventCard(
            id=ev["id"], title=ev.get("title",""), title_zh=ev.get("title_zh",""),
            summary_zh=ev.get("summary_zh",""), url=ev.get("url",""),
            date=ev.get("date",""), source_hint=ev.get("source_hint",""),
            region=ev.get("region","global"), tags=ev.get("tags",[]) or []
        ))
    brief = ""
    try:
        brief = llm_daily_brief(model, top_cards, kpis)
    except Exception as e:
        print(f"[digest] llm daily brief failed: {e}", flush=True)
    if not brief:
        brief = "今日要点生成失败或信息不足。建议检查信息源抓取与 OpenAI 调用状态。"
    # events already prepared; keep as-is
    return SectionOut(
        id="top",
        name="Top 今日要点",
        brief_zh=brief,
        brief_cn=brief,
        brief_global=brief,
        tags=["结论优先","跨板块汇总"],
        events=top_events
    )

def build_digest(cfg: Dict[str, Any], *, date_str: str, model: str, timeout_http: Tuple[int,int]) -> Dict[str, Any]:
    print("[digest] boot", flush=True)
    print(f"[digest] date={date_str} model={model} limit_raw={cfg.get('limit_raw')} thr={cfg.get('cluster_threshold')} items={cfg.get('items_per_section')}", flush=True)

    # KPIs
    print(f"[digest] kpi start: {len(cfg.get('kpis',[]) or [])}", flush=True)
    kpis = fetch_kpis(cfg, timeout=timeout_http)
    print(f"[digest] kpi done: {len((kpis.get('macro') or [])) + len((kpis.get('metals') or []))}", flush=True)

    # Sections
    sections_cfg = cfg.get("sections", []) or []
    print(f"[digest] sections start: {len(sections_cfg)}", flush=True)

    sections_out: List[SectionOut] = []
    for sec in sections_cfg:
        evs, brief = build_events_for_section(sec, cfg, timeout_http=timeout_http, model=model)
        # Convert EventCards to dicts compatible with template
        events_dicts=[]
        for ev in evs:
            events_dicts.append({
                "id": ev.id,
                "title": ev.title,
                "title_zh": ev.title_zh,
                "summary_zh": ev.summary_zh,
                "url": ev.url,
                "date": ev.date,
                "source_hint": ev.source_hint,
                "region": ev.region,  # 'cn' or 'global'
                "tags": ev.tags
            })
        # create view-specific brief: keep same for now (template will choose)
        sec_out = SectionOut(
            id=safe_str(sec.get("id","")),
            name=safe_str(sec.get("name","")),
            brief_zh=brief,
            brief_cn=brief,
            brief_global=brief,
            tags=[],
            events=events_dicts
        )
        sections_out.append(sec_out)

    # add top section first
    top = make_top_section(sections_out, model, kpis, items=min(12, int(cfg.get("items_per_section", 15))))
    all_sections = [top] + sections_out

    print(f"[digest] sections done: {len(all_sections)}", flush=True)

    digest = {
        "date": date_str,
        "generated_at_bjt": now_bjt().strftime("%Y-%m-%d %H:%M"),
        "sections": [dataclasses.asdict(s) for s in all_sections],
        "kpis": kpis,
        "config": {
            "items_per_section": int(cfg.get("items_per_section", 15)),
            "limit_raw": int(cfg.get("limit_raw", 25)),
            "cluster_threshold": float(cfg.get("cluster_threshold", 0.82)),
        }
    }
    return digest


# ----------------------------
# Render HTML (supports both template styles)
# ----------------------------
def render_html(template_path: str, digest: Dict[str, Any]) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")
    date_str = safe_str(digest.get("date",""))
    # Replace JSON placeholder if present
    if "__DIGEST_JSON__" in tpl:
        js = json.dumps(digest, ensure_ascii=False)
        tpl = tpl.replace("__DIGEST_JSON__", js)
    # Date placeholder
    tpl = tpl.replace("{{DATE}}", date_str)

    # If template includes Jinja syntax beyond {{DATE}}, render with Jinja2
    if Template is not None and ("{%" in tpl or "{{" in tpl):
        try:
            jtpl = Template(tpl)
            # provide both lowercase and uppercase keys for compatibility
            ctx = {
                "DATE": date_str,
                "digest": digest,
                "DIGEST": digest,
                "views": {},
                "VIEWS": {},
            }
            return jtpl.render(**ctx)
        except Exception:
            # fall back to plain
            return tpl
    return tpl


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--date", default="")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=None)
    ap.add_argument("--cluster_threshold", type=float, default=None)
    ap.add_argument("--items_per_section", type=int, default=None)  # accept to avoid unrecognized arg
    ap.add_argument("--http_timeout_connect", type=int, default=10)
    ap.add_argument("--http_timeout_read", type=int, default=25)
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    # allow CLI overrides
    if args.limit_raw is not None:
        cfg["limit_raw"] = args.limit_raw
    if args.cluster_threshold is not None:
        cfg["cluster_threshold"] = args.cluster_threshold
    if args.items_per_section is not None:
        cfg["items_per_section"] = args.items_per_section

    if args.date:
        date_str = args.date
    else:
        date_str = iso_date(now_bjt().date())

    timeout_http = (int(args.http_timeout_connect), int(args.http_timeout_read))

    digest = build_digest(cfg, date_str=date_str, model=args.model, timeout_http=timeout_http)
    print("[digest] render html", flush=True)
    html_out = render_html(args.template, digest)
    Path(args.out).write_text(html_out, encoding="utf-8")
    print(f"[digest] wrote {args.out} bytes={len(html_out)}", flush=True)


if __name__ == "__main__":
    main()
