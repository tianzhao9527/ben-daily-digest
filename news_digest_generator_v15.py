#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_generator_v15.py (COMPLETE FIX v2)

修复内容：
1. 增强OpenAI JSON解析，修复JSONDecodeError
2. 添加中国专属新闻查询 (每个板块)
3. 完善区域检测 (中国关键词匹配)
4. 所有标题和摘要强制翻译成中文
5. brief_zh包含决策建议
6. KPI颜色与趋势线匹配
7. 增加反馈数据结构支持
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

@dataclasses.dataclass
class Event:
    title_zh: str
    summary_zh: str
    region: str = "global"
    score: float = 0.5
    sources: list = dataclasses.field(default_factory=list)

# -------------------------
# 中国相关检测 (增强版)
# -------------------------

CHINA_KEYWORDS = [
    # 英文关键词
    "china", "chinese", "beijing", "shanghai", "shenzhen", "guangzhou", 
    "hong kong", "hongkong", "taiwan", "taipei", "macau",
    "prc", "ccp", "xi jinping", "pboc", "cny", "rmb", "renminbi", "yuan",
    "huawei", "alibaba", "tencent", "bytedance", "baidu", "xiaomi", "jd.com",
    "byd", "catl", "nio", "xpeng", "li auto", "geely", "great wall",
    "sinopec", "petrochina", "cnooc", "china mobile", "china telecom",
    "icbc", "bank of china", "ccb", "agricultural bank",
    "zhongguancun", "pudong", "hainan", "xinjiang", "tibet",
    "belt and road", "bri", "aiib", "rcep",
    "made in china", "china manufacturing", "chinese exports",
    "us-china", "sino-", "china trade", "china tariff",
    "smic", "ymtc", "cxmt", "cambricon",
    # 中文关键词
    "中国", "中共", "北京", "上海", "深圳", "广州", "香港", "台湾", "澳门",
    "人民币", "央行", "国务院", "发改委", "工信部", "商务部",
    "华为", "阿里", "腾讯", "字节", "百度", "小米", "比亚迪", "宁德时代",
    "一带一路", "双循环", "内循环", "碳中和", "碳达峰",
]

def guess_region(text: str) -> str:
    """检测文本是否与中国相关"""
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
# 中国专属查询 (每个板块)
# -------------------------

CHINA_QUERIES = {
    "macro": [
        "China PBOC interest rate policy 2025",
        "人民币 汇率 央行",
        "China monetary policy stimulus",
        "中国 经济 GDP 增长",
    ],
    "sanctions": [
        "China US sanctions export controls semiconductor",
        "中国 制裁 芯片 出口管制",
        "Huawei sanctions chip restriction",
        "China entity list BIS OFAC",
    ],
    "compute": [
        "China AI chip domestic production",
        "中国 算力 芯片 国产",
        "Huawei Ascend GPU AI",
        "China data center AI infrastructure",
    ],
    "metals": [
        "China copper aluminum import demand",
        "中国 有色金属 进口 铜 铝",
        "China LME inventory warehouse",
        "中国 稀土 出口 锂",
    ],
    "carbon": [
        "China carbon market ETS trading",
        "中国 碳市场 碳交易 碳排放",
        "China CBAM response EU",
        "中国 碳中和 碳达峰",
    ],
    "sea": [
        "China Vietnam manufacturing supply chain",
        "中国 东南亚 供应链 转移",
        "China Indonesia nickel investment",
        "Chinese companies Southeast Asia relocation",
    ],
    "frontier": [
        "China quantum computing breakthrough",
        "中国 量子计算 超算",
        "China fusion energy EAST",
        "中国 科技 突破 研发",
    ],
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

# -------------------------
# FRED KPI
# -------------------------

def fred_csv(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def fetch_fred_latest(series_id: str) -> tuple[float | None, str | None]:
    url = fred_csv(series_id)
    def _do():
        r = requests.get(url, timeout=(6, 20))
        r.raise_for_status()
        return r.text
    csv_text = retry(_do, name=f"fred {series_id}", tries=3, base_sleep=1.0)
    
    lines = [x.strip() for x in csv_text.splitlines() if x.strip()]
    for row in reversed(lines[1:]):
        parts = row.split(",")
        if len(parts) < 2:
            continue
        dt, val = parts[0].strip(), parts[1].strip()
        if val in (".", ""):
            continue
        try:
            return float(val), dt
        except Exception:
            continue
    return None, None

def fetch_fred_recent(series_id: str, max_points: int = 40) -> tuple[list[float], list[str]]:
    url = fred_csv(series_id)
    def _do():
        r = requests.get(url, timeout=(6, 20))
        r.raise_for_status()
        return r.text
    csv_text = retry(_do, name=f"fred {series_id} recent", tries=3, base_sleep=1.0)

    lines = [x.strip() for x in csv_text.splitlines() if x.strip()]
    if len(lines) < 2:
        return [], []
    vals: list[float] = []
    dates: list[str] = []
    for row in lines[1:]:
        parts = row.split(",")
        if len(parts) < 2:
            continue
        dt, val_s = parts[0].strip(), parts[1].strip()
        if val_s in (".", ""):
            continue
        try:
            v = float(val_s)
        except Exception:
            continue
        dates.append(dt)
        vals.append(v)
    if not vals:
        return [], []
    if max_points and len(vals) > max_points:
        dates = dates[-max_points:]
        vals = vals[-max_points:]
    return vals, dates

def _color_for_direction(direction: str) -> str:
    d = (direction or "").lower()
    if d == "up":
        return "#16A34A"  # green
    if d == "down":
        return "#DC2626"  # red
    return "#64748B"  # gray

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
    return f'<svg class="spark" viewBox="0 0 100 44" preserveAspectRatio="none" aria-hidden="true"><path d="{d}" stroke="{stroke}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'

# -------------------------
# OpenAI JSON解析 (增强版 - 修复JSONDecodeError)
# -------------------------

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def _openai_headers() -> dict:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def fix_json_string(s: str) -> str:
    """修复常见的JSON格式问题"""
    # 移除markdown代码块
    s = re.sub(r'^```json\s*', '', s.strip())
    s = re.sub(r'^```\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    s = s.strip()
    
    # 修复缺少逗号: } { -> },{
    s = re.sub(r'\}\s*\{', '},{', s)
    # 修复缺少逗号: } " -> },"
    s = re.sub(r'\}\s*"', '},"', s)
    # 修复缺少逗号: ] [ -> ],[
    s = re.sub(r'\]\s*\[', '],[', s)
    # 修复缺少逗号: ] " -> ],"
    s = re.sub(r'\]\s*"', '],"', s)
    
    # 修复 "value"\n"key": 格式 (最常见的LLM错误)
    s = re.sub(r'"\s*\n\s*"([^"]+)":', r'","\1":', s)
    
    # 修复数字/布尔后面缺少逗号
    s = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', s)
    s = re.sub(r'(true|false|null)\s*\n\s*"', r'\1,\n"', s)
    
    # 移除尾部逗号
    s = re.sub(r',\s*\}', '}', s)
    s = re.sub(r',\s*\]', ']', s)
    
    return s

def extract_first_json_object(text: str) -> dict | None:
    """增强的JSON提取 - 多策略"""
    
    # 策略1: 直接解析
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    
    # 策略2: 修复后解析
    fixed = fix_json_string(text)
    try:
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    
    # 策略3: 提取JSON对象边界
    start = text.find("{")
    if start < 0:
        return None
    
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
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i+1]
                    cand = fix_json_string(cand)
                    try:
                        obj = json.loads(cand)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        pass
                    return None
    return None

def openai_chat_json(model: str, messages: list[dict], *, timeout=(10, 90), max_tokens: int = 2000) -> dict:
    """调用OpenAI并返回JSON对象"""
    payload_base = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    def _post(payload: dict) -> requests.Response:
        r = requests.post(OPENAI_URL, headers=_openai_headers(), data=json.dumps(payload), timeout=timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.text[:400]}", response=r)
        return r

    # 尝试1: 使用response_format
    try_payload = dict(payload_base)
    try_payload["response_format"] = {"type": "json_object"}

    try:
        resp = retry(lambda: _post(try_payload), name="openai", tries=3, base_sleep=2.0)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        obj = extract_first_json_object(content)
        if obj is None:
            raise ValueError("OpenAI did not return JSON object")
        return obj
    except Exception as e1:
        log(f"[digest] openai json_mode failed: {e1}, trying fallback...")
        
        # 尝试2: 不使用response_format
        try:
            resp = retry(lambda: _post(payload_base), name="openai(no_rf)", tries=2, base_sleep=2.0)
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            obj = extract_first_json_object(content)
            if obj is None:
                raise ValueError(f"OpenAI did not return JSON object (fallback)")
            return obj
        except Exception as e2:
            raise ValueError(f"OpenAI all attempts failed: {e1} / {e2}")

# -------------------------
# LLM prompts (增强版 - 决策建议 + 中文翻译)
# -------------------------

def llm_section_pack(model: str, section: Section, raw_items: list[RawItem], *, items_per_section: int) -> tuple[str, list[Event]]:
    """使用LLM生成中文摘要和事件卡"""
    
    # 准备输入数据，标注中国相关
    lines = []
    cn_items = []
    for it in raw_items[: min(len(raw_items), max(18, items_per_section * 2))]:
        item_data = {
            "title": it.title,
            "source": it.source,
            "date": it.published_at,
            "url": it.url,
            "is_china_related": it.region == "cn",
        }
        lines.append(item_data)
        if it.region == "cn":
            cn_items.append(item_data)

    sys_prompt = """你是Ben的专业金融新闻分析助手。你的任务是：
1. 生成一段中文综合摘要 (brief_zh): 300-500字
2. 生成多个事件卡 (events): 每个包含中文标题和摘要

【重要要求】
- 必须返回纯JSON对象，不要markdown代码块，不要```
- 所有标题(title_zh)和摘要(summary_zh)必须是中文
- 英文新闻必须翻译成中文
- brief_zh必须包含"决策建议"部分
- 优先选择中国相关的新闻(is_china_related=true)
- 如果有中国相关新闻，至少选择2-3条"""

    user_prompt = {
        "section": {"id": section.id, "name": section.name},
        "items": lines,
        "china_related_items": cn_items[:5],
        "output_format": {
            "brief_zh": "300-500字中文摘要，结构：【今日主线】xxx 【关键变化】xxx 【决策建议】xxx",
            "events": [
                {
                    "title_zh": "中文标题(不超过22字)",
                    "summary_zh": "中文摘要(100-150字，包含背景、影响、后续关注点)",
                    "region": "cn或us或eu或asia或global",
                    "score": "0-1重要性评分(中国相关的给更高分)",
                    "sources": [{"title": "原标题", "url": "链接", "source": "来源", "date": "日期"}]
                }
            ]
        },
        "requirements": f"生成{items_per_section}个events。如果有中国相关新闻(is_china_related=true)，必须优先选择并将region设为cn。"
    }

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]
    
    obj = openai_chat_json(model, messages, max_tokens=2500)

    brief_zh = (obj.get("brief_zh") or obj.get("brief") or "").strip()
    evs = obj.get("events") or []
    events: list[Event] = []
    
    if isinstance(evs, list):
        for e in evs[:items_per_section]:
            try:
                title_zh = str(e.get("title_zh") or e.get("title") or "").strip()
                summary_zh = str(e.get("summary_zh") or e.get("summary") or "").strip()
                region = str(e.get("region") or "global").strip().lower()
                
                # 验证region
                if region not in ("cn", "us", "eu", "asia", "global"):
                    region = guess_region(title_zh + " " + summary_zh)
                
                score = float(e.get("score") if e.get("score") is not None else 0.5)
                sources = e.get("sources") if isinstance(e.get("sources"), list) else []
                sources2 = []
                for s in sources[:4]:
                    if not isinstance(s, dict):
                        continue
                    sources2.append({
                        "title": str(s.get("title") or "")[:140],
                        "url": str(s.get("url") or ""),
                        "source": str(s.get("source") or ""),
                        "published_at": str(s.get("date") or s.get("published_at") or ""),
                    })
                if title_zh and summary_zh:
                    events.append(Event(title_zh=title_zh, summary_zh=summary_zh, region=region, score=score, sources=sources2))
            except Exception:
                continue

    if len(events) < max(3, items_per_section // 3):
        raise ValueError(f"LLM returned too few events: {len(events)}")
    
    return brief_zh, events

# -------------------------
# Fallback (确保页面不会空白)
# -------------------------

def fallback_pack(section: Section, raw_items: list[RawItem], *, items_per_section: int) -> tuple[str, list[Event]]:
    """当LLM失败时的fallback"""
    # 优先选择中国相关的
    cn_items = [it for it in raw_items if it.region == "cn"]
    other_items = [it for it in raw_items if it.region != "cn"]
    
    # 混合选择：先中国相关，再其他
    picked = cn_items[:items_per_section//2] + other_items[:items_per_section - len(cn_items[:items_per_section//2])]
    picked = picked[:items_per_section]
    
    events = []
    for it in picked:
        title = it.title.strip()
        summary = f"{it.source} 报道：{title}"
        events.append(Event(
            title_zh=title[:30],
            summary_zh=summary[:180],
            region=it.region,
            score=0.5 if it.region == "cn" else 0.4,
            sources=[{"title": it.title[:140], "url": it.url, "source": it.source, "published_at": it.published_at or ""}],
        ))
    
    cn_count = sum(1 for e in events if e.region == "cn")
    brief = f"""【今日主线】本板块抓取到 {len(raw_items)} 条资讯，其中中国相关 {cn_count} 条。
【关键变化】LLM摘要暂时不可用，采用标题级展示。
【决策建议】建议优先关注标记为中国相关(CN)的条目，这些可能与您的业务更相关。"""
    
    return brief, events

# -------------------------
# Digest build / render
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

def build_digest(cfg: dict, *, date: str, model: str, limit_raw: int, items_per_section: int) -> dict:
    # KPIs
    kpis: list[dict] = []
    kpi_cfg = cfg.get("kpis") or []
    log(f"[digest] kpi start: {len(kpi_cfg)}")
    
    for k in kpi_cfg:
        try:
            # 兼容多种配置格式
            series = (k.get("series") or k.get("id") or "").strip()
            title = (k.get("title") or k.get("name") or k.get("label") or series or "KPI").strip()
            unit = (k.get("unit") or "").strip()
            lookback = int(k.get("lookback") or 40)

            values: list[float] = []
            dates: list[str] = []
            if series:
                values, dates = fetch_fred_recent(series, max_points=lookback)

            val = values[-1] if values else None
            updated = dates[-1] if dates else None

            delta = None
            delta_pct = None
            direction = None
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

            # 颜色与趋势线匹配
            color = _color_for_direction(direction)
            spark = _spark_svg(values[-min(len(values), 40):], stroke=color) if values else ""

            kpis.append({
                "id": series or title,
                "name": title,
                "unit": unit,
                "value": val,
                "updated_at": updated,
                "source": "FRED" if series else (k.get("source") or ""),
                "delta": delta,
                "delta_pct": delta_pct,
                "direction": direction,
                "spark_svg": spark,
                "color": color,  # 颜色与趋势线匹配
                "series_values": values[-min(len(values), 40):] if values else [],
            })
        except Exception as e:
            log(f"[digest] kpi error {k}: {e}")
            series = (k.get("series") or k.get("id") or "").strip()
            title = (k.get("title") or k.get("name") or series or "KPI").strip()
            kpis.append({
                "id": series or title,
                "name": title,
                "unit": (k.get("unit") or "").strip(),
                "value": None,
                "updated_at": None,
                "source": "FRED",
                "delta": None,
                "delta_pct": None,
                "direction": None,
                "spark_svg": "",
                "color": "#64748B",
                "series_values": [],
            })

    log(f"[digest] kpi done: {len(kpis)}")

    sections_cfg = normalize_sections_cfg(cfg.get("sections") or [])
    log(f"[digest] sections start: {len(sections_cfg)}")

    sections_out: list[dict] = []
    for sc in sections_cfg:
        sid = sc.get("id")
        name = sc.get("name") or sid
        sec = Section(id=sid, name=name, tags=sc.get("tags") or [])
        log(f"[digest] section start: {sid} ({name})")

        raw_items: list[RawItem] = []
        
        # 1. 标准查询 (英文)
        for q in (sc.get("gnews_queries") or sc.get("queries") or []):
            try:
                items = fetch_google_news_items(q, limit=limit_raw)
                log(f"[digest] gnews {sid} q='{q[:24]}...' entries={len(items)}")
                raw_items.extend(items)
            except Exception as e:
                log(f"[digest] gnews {sid} failed: {type(e).__name__}: {e}")
        
        # 2. 中国专属查询 (新增 - 每个板块都有)
        china_queries = CHINA_QUERIES.get(sid, [])
        for q in china_queries[:3]:  # 每个section最多3个中国查询
            try:
                # 使用中文Google News
                items = fetch_google_news_items(q, limit=12, hl="zh-CN", gl="CN", ceid="CN:zh-Hans")
                log(f"[digest] gnews-cn {sid} q='{q[:20]}...' entries={len(items)}")
                # 强制标记为中国相关
                for it in items:
                    it.region = "cn"
                raw_items.extend(items)
            except Exception as e:
                log(f"[digest] gnews-cn {sid} failed: {type(e).__name__}: {e}")

        raw_items = dedupe_items(raw_items)
        raw_items = raw_items[: max(items_per_section * 3, limit_raw)]
        
        # 统计中国相关数量
        cn_count = sum(1 for it in raw_items if it.region == "cn")
        log(f"[digest] {sid} total={len(raw_items)} cn_related={cn_count}")

        # Build section pack
        brief_zh = ""
        events: list[Event] = []
        if raw_items:
            try:
                brief_zh, events = llm_section_pack(model, sec, raw_items, items_per_section=items_per_section)
                log(f"[digest] llm pack success ({sid}): brief={len(brief_zh)}chars, events={len(events)}")
            except Exception as e:
                log(f"[digest] llm pack failed ({sid}): {type(e).__name__}: {e}")
                traceback.print_exc()
                brief_zh, events = fallback_pack(sec, raw_items, items_per_section=items_per_section)
        else:
            brief_zh = "（今日该板块暂无可用内容：无候选来源。）"
            events = []

        # 统计events中的中国相关
        cn_events = sum(1 for e in events if e.region == "cn")

        sections_out.append({
            "id": sec.id,
            "name": sec.name,
            "tags": sec.tags,
            "brief_zh": brief_zh,
            "brief_cn": brief_zh,
            "brief_us": brief_zh,
            "brief_eu": brief_zh,
            "brief_asia": brief_zh,
            "brief_global": brief_zh,
            "events": [dataclasses.asdict(e) for e in events],
            "cn_count": cn_events,
            "total_count": len(events),
        })
        log(f"[digest] section done: {sid} events={len(events)} cn_events={cn_events}")

    log(f"[digest] sections done: {len(sections_out)}")

    # 生成配置摘要 (用于设置面板)
    config_summary = {
        "model": model,
        "limit_raw": limit_raw,
        "items_per_section": items_per_section,
        "kpi_count": len(kpis),
        "section_count": len(sections_out),
        "data_sources": {
            "kpi": "FRED (Federal Reserve Economic Data)",
            "news": "Google News RSS",
            "news_regions": ["en-US", "zh-CN"],
        },
        "china_queries_enabled": True,
        "generator": "news_digest_generator_v15.py (FIXED)",
    }

    digest = {
        "date": date,
        "generated_at_utc": now_utc().isoformat(),
        "generated_at_bjt": now_utc().astimezone(_dt.timezone(_dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M (BJT)"),
        "kpis": kpis,
        "sections": sections_out,
        "meta": config_summary,
    }
    return digest

def render_html(template_path: str, digest: dict) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")
    t = Template(tpl)
    context = {
        "DATE": digest.get("date"),
        "TITLE": f"Ben Daily Digest · {digest.get('date')}",
        "DIGEST_JSON": safe_json_dumps(digest),
    }
    return t.render(**context)

def install_signal_handlers():
    faulthandler.enable(all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        log("[digest] faulthandler registered on SIGUSR1")
    except Exception:
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
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit_raw", type=int, default=25)
    ap.add_argument("--items_per_section", type=int, default=15)
    args = ap.parse_args()

    log("[digest] boot v15-FIXED")
    date = args.date or now_utc().date().isoformat()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    log(f"[digest] date={date} model={args.model} limit_raw={args.limit_raw} items={args.items_per_section}")

    digest = build_digest(cfg, date=date, model=args.model, limit_raw=args.limit_raw, items_per_section=args.items_per_section)
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
