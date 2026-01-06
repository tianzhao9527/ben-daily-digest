#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py (v2 - DeepSeek + arXiv + Top5)

修复内容：
1. 使用DeepSeek API替代OpenAI (需要设置 DEEPSEEK_API_KEY)
2. 支持arXiv论文抓取
3. 新增Top 5重要新闻板块
4. 所有内容强制翻译成中文
5. 每个板块生成500字以内结构化总结
6. 支持重要性系数配置
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
from typing import Any, Dict, List, Optional, Tuple

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
    series: str = ""
    title: str = ""
    unit: str = ""
    source: str = "FRED"
    lookback: int = 40

@dataclasses.dataclass
class Section:
    id: str
    name: str
    tags: list = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class RawItem:
    title: str
    url: str
    source: str = ""
    published_at: str = ""
    region: str = "global"
    abstract: str = ""  # For arXiv

@dataclasses.dataclass
class Event:
    title_zh: str
    summary_zh: str
    region: str = "global"
    score: float = 0.5
    sources: list = dataclasses.field(default_factory=list)

# -------------------------
# 重要性系数 (可配置)
# -------------------------

DEFAULT_IMPORTANCE_WEIGHTS = {
    "china_related": 1.5,      # 中国相关新闻权重
    "source_tier1": 1.3,       # 一级来源 (Bloomberg, Reuters, FT)
    "source_tier2": 1.1,       # 二级来源 (WSJ, SCMP)
    "recency_24h": 1.2,        # 24小时内
    "recency_48h": 1.0,        # 48小时内
    "keyword_match": 1.2,      # 关键词匹配
}

TIER1_SOURCES = ["bloomberg", "reuters", "financial times", "ft.com", "wsj", "wall street"]
TIER2_SOURCES = ["scmp", "nikkei", "economist", "cnn", "bbc", "cnbc"]

def calculate_importance(item: RawItem, weights: dict) -> float:
    """计算新闻重要性分数"""
    score = 1.0
    title_lower = item.title.lower()
    source_lower = item.source.lower()
    
    # 中国相关
    if item.region == "cn":
        score *= weights.get("china_related", 1.5)
    
    # 来源等级
    if any(s in source_lower for s in TIER1_SOURCES):
        score *= weights.get("source_tier1", 1.3)
    elif any(s in source_lower for s in TIER2_SOURCES):
        score *= weights.get("source_tier2", 1.1)
    
    # 时效性加权：越新越重要
    if item.published_at:
        try:
            pub_date = _dt.datetime.strptime(item.published_at, "%Y-%m-%d")
            pub_date = pub_date.replace(tzinfo=_dt.timezone.utc)
            days_ago = (now_utc() - pub_date).total_seconds() / 86400
            if days_ago <= 1:  # 24小时内
                score *= weights.get("recency_24h", 1.3)
            elif days_ago <= 2:  # 48小时内
                score *= weights.get("recency_48h", 1.1)
            # 超过2天的不加权
        except:
            pass
    
    return min(score, 2.5)  # 上限2.5

# -------------------------
# 中国相关检测
# -------------------------

CHINA_KEYWORDS = [
    "china", "chinese", "beijing", "shanghai", "shenzhen", "guangzhou", 
    "hong kong", "hongkong", "taiwan", "taipei", "macau",
    "prc", "ccp", "xi jinping", "pboc", "cny", "rmb", "renminbi", "yuan",
    "huawei", "alibaba", "tencent", "bytedance", "baidu", "xiaomi",
    "byd", "catl", "nio", "xpeng", "li auto", "geely",
    "sinopec", "petrochina", "cnooc", "china mobile",
    "belt and road", "bri", "aiib", "rcep",
    "us-china", "sino-", "china trade", "china tariff",
    "smic", "ymtc", "cxmt", "cambricon",
    "中国", "中共", "北京", "上海", "深圳", "广州", "香港", "台湾",
    "人民币", "央行", "华为", "阿里", "腾讯", "比亚迪",
]

def guess_region(text: str) -> str:
    if not text:
        return "global"
    t = text.lower()
    for kw in CHINA_KEYWORDS:
        if kw.lower() in t:
            return "cn"
    if re.search(r'[\u4e00-\u9fff]', text):
        return "cn"
    return "global"

# -------------------------
# 中国专属查询
# -------------------------

CHINA_QUERIES = {
    "macro": ["China PBOC interest rate", "人民币 汇率 央行", "China economy GDP"],
    "sanctions": ["China US sanctions semiconductor", "中国 制裁 芯片", "Huawei sanctions"],
    "compute": ["China AI chip domestic", "中国 算力 芯片", "Huawei Ascend GPU"],
    "metals": ["China copper aluminum import", "中国 有色金属 进口", "China LME"],
    "carbon": ["China carbon market ETS", "中国 碳市场 碳交易", "China CBAM"],
    "sea": ["China Vietnam manufacturing", "中国 东南亚 供应链", "China Indonesia nickel"],
    "frontier": ["China quantum computing", "中国 量子计算", "China fusion energy"],
}

# -------------------------
# Google News RSS
# -------------------------

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"

def gnews_rss_url(query: str, hl: str = "en-US", gl: str = "US", ceid: str = "US:en") -> str:
    q = urllib.parse.quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

def parse_rss_items(xml_text: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "").strip()
            if title and link:
                items.append({"title": title, "link": link, "pubDate": pub, "source": source})
    except Exception:
        pass
    return items

def fetch_google_news_items(query: str, *, limit: int = 25, hl: str = "en-US", gl: str = "US", ceid: str = "US:en") -> list[RawItem]:
    url = gnews_rss_url(query, hl=hl, gl=gl, ceid=ceid)
    def _do():
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(8, 25))
        r.raise_for_status()
        return r.text
    xml_text = retry(_do, name=f"gnews:{query[:20]}", tries=3, base_sleep=1.0)
    
    items = parse_rss_items(xml_text)
    out = []
    for it in items[:limit]:
        title = it.get("title", "")
        region = guess_region(title)
        dt_str = ""
        dt_obj = parse_rfc2822(it.get("pubDate", ""))
        if dt_obj:
            dt_str = dt_obj.strftime("%Y-%m-%d")
        out.append(RawItem(
            title=title,
            url=it.get("link", ""),
            source=it.get("source", ""),
            published_at=dt_str,
            region=region,
        ))
    return out

# -------------------------
# arXiv RSS
# -------------------------

def fetch_arxiv_items(query: str, *, limit: int = 20) -> list[RawItem]:
    """抓取arXiv论文"""
    # arXiv API
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    def _do():
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(10, 30))
        r.raise_for_status()
        return r.text
    
    try:
        xml_text = retry(_do, name="arxiv", tries=3, base_sleep=1.0)
    except Exception as e:
        log(f"[digest] arxiv fetch failed: {e}")
        return []
    
    items = []
    try:
        # Parse Atom feed
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        for entry in root.findall(".//atom:entry", ns):
            title = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
            link_el = entry.find("atom:id", ns)
            link = link_el.text if link_el is not None else ""
            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()[:500]
            published = (entry.findtext("atom:published", namespaces=ns) or "")[:10]
            
            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.findtext("atom:name", namespaces=ns)
                if name:
                    authors.append(name)
            source = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            
            if title and link:
                items.append(RawItem(
                    title=title,
                    url=link,
                    source=source or "arXiv",
                    published_at=published,
                    region="global",
                    abstract=summary,
                ))
    except Exception as e:
        log(f"[digest] arxiv parse error: {e}")
    
    return items

def dedupe_items(items: list[RawItem]) -> list[RawItem]:
    seen = set()
    out = []
    for it in items:
        key = sha1(it.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def filter_by_recency(items: list[RawItem], max_days: int = 7) -> list[RawItem]:
    """过滤只保留最近N天内的新闻"""
    now = now_utc()
    cutoff = now - _dt.timedelta(days=max_days)
    
    out = []
    filtered_count = 0
    for it in items:
        if not it.published_at:
            # 没有日期的新闻，假设是最近的
            out.append(it)
            continue
        try:
            # 解析日期 (格式: YYYY-MM-DD)
            pub_date = _dt.datetime.strptime(it.published_at, "%Y-%m-%d")
            pub_date = pub_date.replace(tzinfo=_dt.timezone.utc)
            if pub_date >= cutoff:
                out.append(it)
            else:
                filtered_count += 1
        except:
            # 解析失败，保留
            out.append(it)
    
    if filtered_count > 0:
        log(f"[digest] recency filter: removed {filtered_count} old items (>{max_days} days)")
    
    return out

# -------------------------
# FRED KPI
# -------------------------

def fred_csv(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def fetch_fred_recent(series_id: str, max_points: int = 40) -> tuple[list[float], list[str]]:
    url = fred_csv(series_id)
    def _do():
        r = requests.get(url, timeout=(6, 20))
        r.raise_for_status()
        return r.text
    csv_text = retry(_do, name=f"fred {series_id}", tries=3, base_sleep=1.0)

    lines = [x.strip() for x in csv_text.splitlines() if x.strip()]
    if len(lines) < 2:
        return [], []
    vals, dates = [], []
    for row in lines[1:]:
        parts = row.split(",")
        if len(parts) < 2:
            continue
        dt, val_s = parts[0].strip(), parts[1].strip()
        if val_s in (".", ""):
            continue
        try:
            vals.append(float(val_s))
            dates.append(dt)
        except:
            continue
    if max_points and len(vals) > max_points:
        dates = dates[-max_points:]
        vals = vals[-max_points:]
    return vals, dates

def _color_for_direction(direction: str) -> str:
    d = (direction or "").lower()
    if d == "up":
        return "#34C759"
    if d == "down":
        return "#FF3B30"
    return "#8E8E93"

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
    return f'<svg class="spark" viewBox="0 0 100 44" preserveAspectRatio="none"><path d="{d}" stroke="{stroke}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'

# -------------------------
# DeepSeek API (替代OpenAI)
# -------------------------

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def _deepseek_headers() -> dict:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def fix_json_string(s: str) -> str:
    # 移除markdown代码块
    s = re.sub(r'^```json\s*\n?', '', s.strip())
    s = re.sub(r'^```\s*\n?', '', s)
    s = re.sub(r'\n?```$', '', s)
    s = s.strip()
    
    # 移除可能的前缀文字（如 "以下是JSON："）
    json_start = s.find('{')
    if json_start > 0:
        s = s[json_start:]
    
    # 修复常见格式问题
    s = re.sub(r'\}\s*\{', '},{', s)
    s = re.sub(r'\}\s*"', '},"', s)
    s = re.sub(r'\]\s*\[', '],[', s)
    s = re.sub(r'\]\s*"', '],"', s)
    s = re.sub(r'"\s*\n\s*"([^"]+)":', r'","\1":', s)
    s = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', s)
    s = re.sub(r'(true|false|null)\s*\n\s*"', r'\1,\n"', s)
    s = re.sub(r',\s*\}', '}', s)
    s = re.sub(r',\s*\]', ']', s)
    
    return s

def extract_first_json_object(text: str) -> dict | None:
    if not text:
        return None
        
    # 尝试1: 直接解析
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except:
        pass
    
    # 尝试2: 修复后解析
    fixed = fix_json_string(text)
    try:
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else None
    except:
        pass
    
    # 尝试3: 查找JSON边界
    start = text.find("{")
    if start < 0:
        return None
    
    # 从后往前找最后一个}
    end = text.rfind("}")
    if end <= start:
        return None
    
    candidate = text[start:end+1]
    candidate = fix_json_string(candidate)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except:
        pass
    
    # 尝试4: 逐字符匹配括号
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
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = fix_json_string(text[start:i+1])
                try:
                    obj = json.loads(cand)
                    return obj if isinstance(obj, dict) else None
                except:
                    return None
    return None

def deepseek_chat_json(messages: list[dict], *, timeout=(15, 120), max_tokens: int = 3000) -> dict:
    """调用DeepSeek API"""
    # 不使用response_format，通过prompt强制JSON输出
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    def _post() -> requests.Response:
        r = requests.post(DEEPSEEK_URL, headers=_deepseek_headers(), 
                         data=json.dumps(payload), timeout=timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.text[:400]}", response=r)
        return r

    resp = retry(_post, name="deepseek", tries=3, base_sleep=2.0)
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    log(f"[digest] deepseek raw response length: {len(content)}")
    
    obj = extract_first_json_object(content)
    if obj is None:
        # 记录原始响应以便调试
        log(f"[digest] deepseek non-JSON response: {content[:500]}...")
        raise ValueError("DeepSeek did not return JSON object")
    return obj

# -------------------------
# LLM Section Pack (强制中文翻译+结构化总结)
# -------------------------

def llm_section_pack(section: Section, raw_items: list[RawItem], *, items_per_section: int) -> tuple[str, list[Event]]:
    """使用DeepSeek生成中文摘要和事件卡"""
    
    lines = []
    for it in raw_items[: min(len(raw_items), max(18, items_per_section * 2))]:
        lines.append({
            "title": it.title,
            "source": it.source,
            "date": it.published_at,
            "url": it.url,
            "is_china": it.region == "cn",
            "abstract": it.abstract[:200] if it.abstract else "",
        })

    # 简化prompt，确保JSON输出
    sys_prompt = """你是专业金融新闻分析师。请分析以下新闻并返回JSON格式结果。

【重要】你必须只返回一个JSON对象，不要有任何其他文字、markdown或代码块。

JSON格式：
{"brief_zh":"板块总结(中文,300字,包含【核心动态】【趋势研判】【决策参考】三部分)","events":[{"title_zh":"中文标题(20字内)","summary_zh":"中文摘要(100字)","region":"cn或global","score":0.5,"sources":[{"title":"原标题","url":"链接","source":"来源"}]}]}

要求：
1. 所有英文必须翻译成中文
2. 优先选择中国相关新闻(is_china=true)
3. 只返回JSON，不要任何解释"""

    user_content = f"板块：{section.name}\n新闻列表：\n{json.dumps(lines, ensure_ascii=False)}\n\n请生成{items_per_section}条events，只返回JSON："

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
    
    obj = deepseek_chat_json(messages, max_tokens=3500)

    brief_zh = (obj.get("brief_zh") or "").strip()
    evs = obj.get("events") or []
    events: list[Event] = []
    
    if isinstance(evs, list):
        for e in evs[:items_per_section]:
            try:
                title_zh = str(e.get("title_zh") or e.get("title") or "").strip()
                summary_zh = str(e.get("summary_zh") or e.get("summary") or "").strip()
                region = str(e.get("region") or "global").strip().lower()
                if region not in ("cn", "us", "eu", "asia", "global"):
                    region = guess_region(title_zh)
                score = float(e.get("score") if e.get("score") is not None else 0.5)
                sources = e.get("sources") if isinstance(e.get("sources"), list) else []
                sources2 = []
                for s in sources[:4]:
                    if isinstance(s, dict):
                        sources2.append({
                            "title": str(s.get("title") or "")[:140],
                            "url": str(s.get("url") or ""),
                            "source": str(s.get("source") or ""),
                            "published_at": str(s.get("date") or s.get("published_at") or ""),
                        })
                if title_zh and summary_zh:
                    events.append(Event(title_zh=title_zh, summary_zh=summary_zh, region=region, score=score, sources=sources2))
            except:
                continue

    if len(events) < max(2, items_per_section // 4):
        raise ValueError(f"LLM returned too few events: {len(events)}")
    
    return brief_zh, events

# -------------------------
# Top 5 重要新闻生成
# -------------------------

def generate_top5(all_events: list[dict]) -> list[dict]:
    """从所有事件中选出最重要的5条"""
    if not all_events:
        return []
    
    # 按score排序
    sorted_events = sorted(all_events, key=lambda x: x.get("score", 0), reverse=True)
    
    # 取前5条，但确保至少2条中国相关
    top5 = []
    cn_events = [e for e in sorted_events if e.get("region") == "cn"]
    other_events = [e for e in sorted_events if e.get("region") != "cn"]
    
    # 先加2条中国相关
    top5.extend(cn_events[:2])
    # 再加其他高分新闻
    for e in other_events:
        if len(top5) >= 5:
            break
        top5.append(e)
    # 如果还不够5条，继续加中国相关
    for e in cn_events[2:]:
        if len(top5) >= 5:
            break
        top5.append(e)
    
    return top5[:5]

def llm_generate_top5_summary(events: list[dict]) -> list[dict]:
    """为Top 5新闻生成决策导向的总结"""
    if not events:
        return []
    
    # 简化prompt
    sys_prompt = """为以下重要新闻生成中文总结。只返回JSON，格式：
{"top5":[{"title":"中文标题","summary":"总结(150字,说明对决策的影响)","action":"建议行动(30字)"}]}
不要任何其他文字。"""

    items = [{"title": e.get("title_zh", ""), "summary": e.get("summary_zh", "")[:100]} for e in events[:5]]
    user_content = f"新闻：{json.dumps(items, ensure_ascii=False)}\n只返回JSON："

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
    
    try:
        obj = deepseek_chat_json(messages, max_tokens=1500)
        return obj.get("top5", [])
    except Exception as e:
        log(f"[digest] top5 summary failed: {e}")
        # Fallback：直接使用原始数据
        return [{"title": e.get("title_zh", "")[:30], "summary": e.get("summary_zh", "")[:150], "action": "关注后续发展"} for e in events[:5]]

# -------------------------
# Fallback
# -------------------------

def fallback_pack(section: Section, raw_items: list[RawItem], *, items_per_section: int) -> tuple[str, list[Event]]:
    """当LLM失败时的fallback - 尽量保持中文"""
    cn_items = [it for it in raw_items if it.region == "cn"]
    other_items = [it for it in raw_items if it.region != "cn"]
    picked = cn_items[:items_per_section//2] + other_items[:items_per_section - len(cn_items[:items_per_section//2])]
    picked = picked[:items_per_section]
    
    events = []
    for it in picked:
        # 标题：如果已经是中文就保留，否则截断
        title = it.title.strip()
        if len(title) > 50:
            title = title[:47] + "..."
        
        # 摘要：来源+标题
        summary = f"来源 {it.source}：{title}"
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        events.append(Event(
            title_zh=title,
            summary_zh=summary,
            region=it.region,
            score=0.6 if it.region == "cn" else 0.4,
            sources=[{"title": it.title[:100], "url": it.url, "source": it.source, "published_at": it.published_at or ""}],
        ))
    
    cn_count = sum(1 for e in events if e.region == "cn")
    brief = f"""【核心动态】本板块抓取到 {len(raw_items)} 条资讯，其中中国相关 {cn_count} 条。
【趋势研判】LLM翻译暂时不可用，显示原始标题。
【决策参考】建议优先关注中国相关条目，点击来源链接查看详情。"""
    
    return brief, events

# -------------------------
# Digest build
# -------------------------

def normalize_sections_cfg(cfg_sections) -> list[dict]:
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

def build_digest(cfg: dict, *, date: str, limit_raw: int, items_per_section: int) -> dict:
    importance_weights = cfg.get("importance_weights", DEFAULT_IMPORTANCE_WEIGHTS)
    
    # KPIs
    kpis: list[dict] = []
    kpi_cfg = cfg.get("kpis") or []
    log(f"[digest] kpi start: {len(kpi_cfg)}")
    
    for k in kpi_cfg:
        try:
            series = (k.get("series") or k.get("id") or "").strip()
            title = (k.get("title") or k.get("name") or series or "KPI").strip()
            unit = (k.get("unit") or "").strip()
            lookback = int(k.get("lookback") or 40)

            values, dates = [], []
            if series:
                values, dates = fetch_fred_recent(series, max_points=lookback)

            val = values[-1] if values else None
            updated = dates[-1] if dates else None

            delta, delta_pct, direction = None, None, None
            if values and len(values) >= 2 and val is not None:
                prev = values[-2]
                delta = val - prev
                if prev != 0:
                    delta_pct = (delta / prev) * 100.0
                if abs(delta) < 1e-12:
                    direction = "flat"
                elif delta > 0:
                    direction = "up"
                else:
                    direction = "down"

            color = _color_for_direction(direction)
            spark = _spark_svg(values[-min(len(values), 40):], stroke=color) if values else ""

            kpis.append({
                "id": series or title,
                "name": title,
                "unit": unit,
                "value": val,
                "updated_at": updated,
                "source": "FRED",
                "delta": delta,
                "delta_pct": delta_pct,
                "direction": direction,
                "spark_svg": spark,
                "color": color,
                "series_values": values[-min(len(values), 40):] if values else [],
            })
        except Exception as e:
            log(f"[digest] kpi error: {e}")
            series = (k.get("series") or "").strip()
            title = (k.get("title") or k.get("name") or series or "KPI").strip()
            kpis.append({
                "id": series or title, "name": title, "unit": (k.get("unit") or "").strip(),
                "value": None, "updated_at": None, "source": "FRED",
                "delta": None, "delta_pct": None, "direction": None,
                "spark_svg": "", "color": "#8E8E93", "series_values": [],
            })

    log(f"[digest] kpi done: {len(kpis)}")

    sections_cfg = normalize_sections_cfg(cfg.get("sections") or [])
    log(f"[digest] sections start: {len(sections_cfg)}")

    sections_out: list[dict] = []
    all_events_for_top5: list[dict] = []
    
    for sc in sections_cfg:
        sid = sc.get("id")
        name = sc.get("name") or sid
        sec = Section(id=sid, name=name, tags=sc.get("tags") or [])
        log(f"[digest] section start: {sid} ({name})")

        raw_items: list[RawItem] = []
        
        # 检查是否是arXiv类型
        if sc.get("type") == "arxiv":
            arxiv_query = sc.get("arxiv_query", "cat:cs.AI")
            try:
                items = fetch_arxiv_items(arxiv_query, limit=20)
                log(f"[digest] arxiv {sid} entries={len(items)}")
                raw_items.extend(items)
            except Exception as e:
                log(f"[digest] arxiv {sid} failed: {e}")
        else:
            # 标准Google News查询
            for q in (sc.get("gnews_queries") or sc.get("queries") or []):
                try:
                    items = fetch_google_news_items(q, limit=limit_raw)
                    log(f"[digest] gnews {sid} q='{q[:24]}...' entries={len(items)}")
                    raw_items.extend(items)
                except Exception as e:
                    log(f"[digest] gnews {sid} failed: {e}")
            
            # 中国专属查询
            china_queries = CHINA_QUERIES.get(sid, [])
            for q in china_queries[:2]:
                try:
                    items = fetch_google_news_items(q, limit=10, hl="zh-CN", gl="CN", ceid="CN:zh-Hans")
                    log(f"[digest] gnews-cn {sid} q='{q[:20]}...' entries={len(items)}")
                    for it in items:
                        it.region = "cn"
                    raw_items.extend(items)
                except Exception as e:
                    log(f"[digest] gnews-cn {sid} failed: {e}")

        raw_items = dedupe_items(raw_items)
        # 时效性过滤：只保留7天内的新闻（放宽以确保有足够内容）
        raw_items = filter_by_recency(raw_items, max_days=7)
        raw_items = raw_items[: max(items_per_section * 3, limit_raw)]
        
        # 计算重要性分数
        for it in raw_items:
            it_score = calculate_importance(it, importance_weights)
            # 暂存分数用于后续排序
        
        cn_count = sum(1 for it in raw_items if it.region == "cn")
        log(f"[digest] {sid} total={len(raw_items)} cn_related={cn_count}")

        brief_zh = ""
        events: list[Event] = []
        if raw_items:
            try:
                brief_zh, events = llm_section_pack(sec, raw_items, items_per_section=items_per_section)
                log(f"[digest] llm pack success ({sid}): events={len(events)}")
            except Exception as e:
                log(f"[digest] llm pack failed ({sid}): {type(e).__name__}: {e}")
                traceback.print_exc()
                brief_zh, events = fallback_pack(sec, raw_items, items_per_section=items_per_section)
        else:
            brief_zh = "【核心动态】今日该板块暂无可用内容。\n【关键数据】无\n【趋势研判】无\n【决策参考】请关注其他信息源。"
            events = []

        cn_events = sum(1 for e in events if e.region == "cn")
        
        section_data = {
            "id": sec.id,
            "name": sec.name,
            "tags": sec.tags,
            "brief_zh": brief_zh,
            "events": [dataclasses.asdict(e) for e in events],
            "cn_count": cn_events,
            "total_count": len(events),
        }
        sections_out.append(section_data)
        
        # 收集所有事件用于Top 5
        for e in events:
            all_events_for_top5.append({
                "title_zh": e.title_zh,
                "summary_zh": e.summary_zh,
                "region": e.region,
                "score": e.score,
                "section": sec.name,
                "sources": e.sources,
            })
        
        log(f"[digest] section done: {sid} events={len(events)}")

    log(f"[digest] sections done: {len(sections_out)}")

    # 生成Top 5
    log("[digest] generating top 5...")
    top5_events = generate_top5(all_events_for_top5)
    top5_with_summary = llm_generate_top5_summary(top5_events)
    log(f"[digest] top 5 done: {len(top5_with_summary)}")

    config_summary = {
        "llm_provider": "DeepSeek",
        "llm_model": "deepseek-chat",
        "limit_raw": limit_raw,
        "items_per_section": items_per_section,
        "kpi_count": len(kpis),
        "section_count": len(sections_out),
        "importance_weights": importance_weights,
        "china_queries_enabled": True,
        "arxiv_enabled": True,
    }

    digest = {
        "date": date,
        "generated_at_utc": now_utc().isoformat(),
        "generated_at_bjt": now_utc().astimezone(_dt.timezone(_dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M"),
        "kpis": kpis,
        "sections": sections_out,
        "top5": top5_with_summary,
        "meta": config_summary,
    }
    return digest

def render_html(template_path: str, digest: dict) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")
    t = Template(tpl)
    context = {
        "DATE": digest.get("date"),
        "TITLE": f"Ben's Daily Brief · {digest.get('date')}",
        "DIGEST_JSON": safe_json_dumps(digest),
    }
    return t.render(**context)

def install_signal_handlers():
    faulthandler.enable(all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except:
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
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items_per_section", type=int, default=15)
    args = ap.parse_args()

    log("[digest] boot v15-DeepSeek")
    
    # 检查API Key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        log("[digest] WARNING: DEEPSEEK_API_KEY not set!")
        log("[digest] Please set it in GitHub Secrets or environment")
    
    date = args.date or now_utc().date().isoformat()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    log(f"[digest] date={date} limit_raw={args.limit_raw} items={args.items_per_section}")

    digest = build_digest(cfg, date=date, limit_raw=args.limit_raw, items_per_section=args.items_per_section)
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
