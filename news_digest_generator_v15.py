#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""news_digest_generator_v15.py
Apple News 风格、配置驱动、输出单文件静态 HTML（无客户端抓取）
数据：Google News RSS / arXiv API / FRED CSV
中文：配置 OPENAI_API_KEY 后自动翻译+300–500字简报
"""
import argparse, datetime, json, math, os, re
from typing import Any, Dict, List, Optional, Tuple

import requests
import feedparser

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
DEFAULT_DENYLIST = {"reuters.com","www.reuters.com","ft.com","www.ft.com","wsj.com","www.wsj.com","bloomberg.com","www.bloomberg.com","economist.com","www.economist.com"}

def bjt_now_str():
  now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
  return now.strftime('%Y-%m-%d %H:%M')

def domain_of(url: str) -> str:
  m = re.search(r"https?://([^/]+)/", url or "")
  return (m.group(1).lower() if m else "")

def is_denied(url: str, denyset: set) -> bool:
  return domain_of(url) in denyset

def norm_tokens(s: str) -> List[str]:
  s = (s or "").lower()
  s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s).strip()
  return [t for t in s.split() if len(t) >= 2]

def jaccard(a: List[str], b: List[str]) -> float:
  A, B = set(a), set(b)
  if not A or not B: return 0.0
  return len(A & B) / len(A | B)

def decide_region(s: str) -> str:
  s2 = (s or "").lower()
  cn_en = ["china","chinese","beijing","shanghai","cny","yuan","hong kong","taiwan"]
  cn_zh = ["中国","北京","上海","人民币","香港","台湾","中美","中欧"]
  if any(k in s2 for k in cn_en) or any(k in (s or "") for k in cn_zh): return "cn"
  return "global"

def fetch_google_news(query: str, limit: int, denyset: set) -> List[Dict[str, Any]]:
  url = GOOGLE_NEWS_RSS.format(q=requests.utils.quote(query))
  fp = feedparser.parse(url)
  out=[]
  for e in fp.entries[:limit]:
    link = getattr(e, 'link', '')
    if not link or is_denied(link, denyset): continue
    title = getattr(e, 'title', '').strip()
    published = getattr(e, 'published', '')
    source = getattr(getattr(e,'source',None),'title','') if getattr(e,'source',None) else ''
    out.append({'title':title,'url':link,'date':published,'source':source or domain_of(link)})
  return out

def fetch_arxiv(limit: int, cats: List[str]) -> List[Dict[str, Any]]:
  cat_query = "+OR+".join([f"cat:{c}" for c in cats]) if cats else "cat:cs.AI+OR+cat:cs.LG"
  q = f"search_query={cat_query}&sortBy=submittedDate&sortOrder=descending&max_results={limit}"
  url = "http://export.arxiv.org/api/query?" + q
  fp = feedparser.parse(url)
  out=[]
  for e in fp.entries:
    out.append({'title':getattr(e,'title','').replace('\n',' ').strip(),
                'url':getattr(e,'link',''),
                'date':getattr(e,'published','')[:10],
                'source':'arXiv.org',
                'abstract':getattr(e,'summary','').replace('\n',' ').strip()})
  return out

def cluster_items(items: List[Dict[str,Any]], threshold: float, max_events: int) -> List[Dict[str,Any]]:
  clusters=[]
  for it in items:
    toks = norm_tokens(it.get('title',''))
    placed=False
    for c in clusters:
      if jaccard(toks, c['_toks']) >= threshold:
        c['sources'].append({'source':it.get('source',''), 'url':it.get('url','')})
        if len(it.get('title','')) > len(c.get('title','')): c['title'] = it.get('title','')
        placed=True; break
    if not placed:
      clusters.append({'title':it.get('title',''),'_toks':toks,'date':it.get('date',''),
                       'sources':[{'source':it.get('source',''), 'url':it.get('url','')}]})
    if len(clusters) >= max_events: break
  for c in clusters: c.pop('_toks',None)
  return clusters

def spark_svg(values: List[float]) -> str:
  vals=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
  if len(vals)<2: return ''
  mn,mx=min(vals),max(vals)
  pts=[0.5 for _ in vals] if (mx-mn)<1e-12 else [(v-mn)/(mx-mn) for v in vals]
  xs=[2+i*(96/(len(pts)-1)) for i in range(len(pts))]
  ys=[5+(1-p)*34 for p in pts]
  d='M '+' L '.join(f"{xs[i]:.2f},{ys[i]:.2f}" for i in range(len(pts)))
  return f"<svg class='spark' viewBox='0 0 100 44' preserveAspectRatio='none'><path d='{d}' stroke='currentColor' stroke-opacity='0.65' fill='none' stroke-width='2'/></svg>"

def fetch_fred_series(series_id: str, tail: int = 60) -> Tuple[Optional[float], Optional[float], List[float]]:
  url = FRED_CSV.format(sid=series_id)
  r = requests.get(url, timeout=25); r.raise_for_status()
  lines=[ln.strip() for ln in r.text.splitlines() if ln.strip()]
  data=[]
  for ln in lines[1:]:
    parts=ln.split(',')
    if len(parts)<2: continue
    v=parts[1].strip()
    try: fv=float(v)
    except: continue
    if math.isfinite(fv): data.append(fv)
  if not data: return None,None,[]
  last=data[-1]; prev=data[-2] if len(data)>=2 else None
  return last, prev, data[-tail:]

def fmt_value(v: Optional[float]) -> str:
  if v is None or not math.isfinite(v): return '—'
  if abs(v)>=1000: return f"{v:,.0f}"
  if abs(v)>=10: return f"{v:,.2f}"
  return f"{v:.4f}".rstrip('0').rstrip('.')

def pct_change(last: Optional[float], prev: Optional[float]):
  if last is None or prev is None: return None,None
  if not (math.isfinite(last) and math.isfinite(prev)): return None,None
  d = last - prev
  p = (d/prev*100) if prev!=0 else None
  return round(d,4), (round(p,2) if p is not None else None)

def llm_available(): return bool(os.environ.get('OPENAI_API_KEY'))

def openai_chat(messages, model):
  api_key=os.environ['OPENAI_API_KEY']
  url='https://api.openai.com/v1/chat/completions'
  headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'}
  payload={'model':model,'messages':messages,'temperature':0.2}
  r=requests.post(url,headers=headers,data=json.dumps(payload),timeout=90); r.raise_for_status()
  return r.json()['choices'][0]['message']['content'].strip()

def zh_title_and_bullet(title: str, model: str):
  if not llm_available(): return title, '（未配置 OPENAI_API_KEY，暂无法自动中文要点。）'
  prompt=f"""请将英文标题翻译为简洁准确的中文标题，并给出一句中文要点（<=40字，不要解释型废话）。
英文标题：{title}
输出两行：
1) 中文标题
2) 要点：..."""
  out=openai_chat([{'role':'user','content':prompt}],model=model)
  lines=[ln.strip() for ln in out.splitlines() if ln.strip()]
  zh=lines[0] if lines else title
  bullet=''
  for ln in lines[1:]:
    if ln.startswith('要点'):
      bullet=re.sub(r'^要点[:：]\s*','',ln).strip(); break
  if not bullet and len(lines)>=2: bullet=lines[1]
  return zh, bullet

def section_brief(name: str, events: List[Dict[str,Any]], view_hint: str, model: str):
  if not llm_available(): return '（未配置 OPENAI_API_KEY：此处将生成 300–500 字中文简报。当前仅展示事件卡。）'
  items='\n'.join([f"- {e['title_zh']}：{e['summary_zh']}" for e in events[:12]])
  prompt=f"""你是企业级每日简报编辑。基于下面事件卡要点，写一段中文栏目简报（300-500字）。
要求：高密度、可决策、避免解释型废话；写清楚“发生了什么/为什么重要/对经营与合规的影响”，并在末尾给出2条可执行动作（用“动作：”开头）。
栏目：{name}
视图：{view_hint}
事件要点：
{items}
只输出正文。"""
  return openai_chat([{'role':'user','content':prompt}],model=model)

def today_brief(sections, model):
  if not llm_available(): return '（未配置 OPENAI_API_KEY：此处将生成 300–500 字“今日要点”。）'
  bullets=[]
  for sec in sections[:6]:
    if sec.get('events'):
      e=sec['events'][0]; bullets.append(f"- {sec['name']}：{e['title_zh']}（{e['summary_zh']}）")
  prompt=f"""你是总编。根据各栏目头部要点，写一段中文“今日要点”（300-500字）。
要求：不空泛、不解释政策背景；按“宏观→合规→产业→商品→东南亚”的逻辑串联，最后给出3条“今日关注指标/阈值”（用“指标：”开头）。
材料：
{chr(10).join(bullets)}
只输出正文。"""
  return openai_chat([{'role':'user','content':prompt}],model=model)

def _safe_json_load(s: str) -> Optional[Dict[str,Any]]:
  try:
    s2=s.strip()
    # allow fenced blocks
    s2=re.sub(r"^```(?:json)?\s*","",s2)
    s2=re.sub(r"\s*```$","",s2)
    return json.loads(s2)
  except Exception:
    return None

def today_brief_struct(sections, model) -> Tuple[str, str, Optional[Dict[str,Any]]]:
  """Return (plain_text, brief_html, struct_dict). Falls back to today_brief()."""
  if not llm_available():
    txt='（未配置 OPENAI_API_KEY：此处将生成结构化“今日要点”。）'
    return txt, f"<div>{txt}</div>", None

  bullets=[]
  for sec in sections[:8]:
    if sec.get('events'):
      e=sec['events'][0]
      bullets.append(f"- {sec['name']}：{e['title_zh']}（{e['summary_zh']}）")
  prompt=f"""你是企业级晨报总编。请把下面材料整理成**结构化**“今日要点”，并用**严格 JSON**输出（只输出 JSON，不要多余文字）。

字段要求：
{{
  \"one_liner\": \"一句话结论（<=28字）\", 
  \"key_signals\": [\"关键变化1\",\"关键变化2\",\"关键变化3\",\"关键变化4\"],
  \"why_it_matters\": [\"为什么重要1\",\"为什么重要2\",\"为什么重要3\"],
  \"actions\": [\"动作1（可执行、可验证）\",\"动作2\",\"动作3\"],
  \"watch\": [\"指标：X | 阈值：Y | 说明：Z\",\"...\",\"...\"]
}}

写作约束：
- 不写背景科普，不写“业内人士表示/值得关注”等空话。
- 逻辑顺序：宏观/市场 → 制裁/合规 → 产业/算力 → 大宗/金属 → 碳/CBAM → 东南亚供应链。
- watch 里的“阈值”要具体（数字/区间/方向），如果材料缺失可给“方向阈值”（例如“较昨日上行/下行”）。

材料：
{chr(10).join(bullets)}
"""
  raw=openai_chat([{'role':'user','content':prompt}],model=model)
  obj=_safe_json_load(raw)
  if not isinstance(obj,dict):
    txt=today_brief(sections,model)
    return txt, f"<div>{txt}</div>", None

  one=str(obj.get('one_liner','')).strip()
  key_signals=[str(x).strip() for x in (obj.get('key_signals') or []) if str(x).strip()]
  why=[str(x).strip() for x in (obj.get('why_it_matters') or []) if str(x).strip()]
  actions=[str(x).strip() for x in (obj.get('actions') or []) if str(x).strip()]
  watch=[str(x).strip() for x in (obj.get('watch') or []) if str(x).strip()]

  # Plain text (for搜索/导出)
  txt=(f"今日要点：{one}\n"+
       "关键变化："+"；".join(key_signals[:6])+"\n"+
       "为什么重要："+"；".join(why[:4])+"\n"+
       "动作："+"；".join(actions[:4])+"\n"+
       "指标："+"；".join(watch[:4]))

  def li(arr):
    return "".join([f"<li>{re.sub(r'<','&lt;',x)}</li>" for x in arr])

  html=f"""
  <div class='topgrid'>
    <div class='tblock'>
      <div class='tlabel'>一句话结论</div>
      <div class='tmain'>{re.sub(r'<','&lt;',one) or '—'}</div>
      <div class='tlabel' style='margin-top:10px'>关键变化</div>
      <ul class='tlist'>{li(key_signals[:6])}</ul>
      <div class='tlabel' style='margin-top:10px'>为什么重要</div>
      <ul class='tlist'>{li(why[:4])}</ul>
    </div>
    <div class='tblock'>
      <div class='tlabel'>今日动作</div>
      <ul class='tlist'>{li(actions[:4])}</ul>
      <div class='tlabel' style='margin-top:10px'>关注指标/阈值</div>
      <ul class='tlist'>{li(watch[:4])}</ul>
    </div>
  </div>
  """
  return txt, html, obj

def build_kpis(cfg):
  out={'generated_at_bjt':bjt_now_str(),'metals':[],'macro':[]}
  for group in ['metals','macro']:
    for s in cfg.get('kpis',{}).get(group,[]):
      if s.get('provider')!='fred': continue
      sid=s['series_id']
      last,prev,tail=fetch_fred_series(sid,tail=70 if group=='macro' else 60)
      d,p=pct_change(last,prev)
      out[group].append({'label':f"{s.get('label','')}（{sid}）",'unit':s.get('unit',''),
                         'value_str':fmt_value(last),'delta':d,'pct':p,'source':'FRED','spark_svg':spark_svg(tail)})
  return out

def render_html(template_path, digest, out_path):
  tpl=open(template_path,'r',encoding='utf-8').read()
  html=tpl.replace('{{DATE}}',digest['date']).replace('__DIGEST_JSON__',json.dumps(digest,ensure_ascii=False))
  with open(out_path,'w',encoding='utf-8') as f: f.write(html)

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument('--config',required=True)
  ap.add_argument('--template',required=True)
  ap.add_argument('--out',default='public/index.html')
  ap.add_argument('--date',default=None)
  ap.add_argument('--model',default='gpt-4o-mini')
  ap.add_argument('--limit_raw',type=int,default=90)
  ap.add_argument('--cluster_threshold',type=float,default=0.42)
  # Backward/CI friendly: allow overriding per-section count from workflow
  ap.add_argument('--items_per_section',type=int,default=None)
  args=ap.parse_args()

  cfg=json.load(open(args.config,'r',encoding='utf-8'))
  date=args.date or (datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
  denyset=set(DEFAULT_DENYLIST) | set(cfg.get('domain_denylist',[]))

  digest={'date':date,'generated_at_bjt':bjt_now_str(),'config':{'version':cfg.get('version','')},'config_full':cfg,
          'kpis':build_kpis(cfg),'sections':[]}

  for sec in cfg.get('sections',[]):
    sid=sec['id']; name=sec['name']; typ=sec.get('type','google_news_cluster')
    per=int(args.items_per_section if args.items_per_section is not None else sec.get('items_per_section',15))
    qg=sec.get('query_global',''); events=[]
    raw=fetch_google_news(qg,limit=args.limit_raw,denyset=denyset) if qg else []
    clusters=cluster_items(raw,threshold=args.cluster_threshold,max_events=per)
    for i,c in enumerate(clusters[:per]):
      zh,bullet=zh_title_and_bullet(c['title'],model=args.model)
      url=c['sources'][0]['url'] if c.get('sources') else ''
      src=c['sources'][0]['source'] if c.get('sources') else ''
      events.append({'id':f"{sid}_{i+1}",'title':c['title'],'title_zh':zh,'summary_zh':bullet,
                     'source_hint':src,'date':(c.get('date','') or '')[:16],'url':url,'tags':sec.get('tags',[])[:4],
                     'region':decide_region(zh)})
    if typ=='mixed_news_arxiv':
      ar=fetch_arxiv(limit=per,cats=sec.get('arxiv_cats',[]))
      base=len(events)
      for j,e in enumerate(ar[:max(0,per-base)]):
        zh,bullet=zh_title_and_bullet(e['title'],model=args.model)
        events.append({'id':f"{sid}_a{j+1}",'title':e['title'],'title_zh':zh,'summary_zh':bullet,
                       'source_hint':'arXiv.org','date':e.get('date',''),'url':e.get('url',''),
                       'tags':(sec.get('tags',[])[:3]+['论文'])[:4],'region':'global'})

    brief_global=section_brief(name,[e for e in events if e['region']=='global'] or events,'global',args.model)
    brief_cn=section_brief(name,[e for e in events if e['region']=='cn'] or events,'cn',args.model)
    digest['sections'].append({'id':sid,'name':name,'tags':sec.get('tags',[]),'brief_zh':brief_global,
                              'brief_global':brief_global,'brief_cn':brief_cn,'events':events})

  if digest['sections']:
    # 1) Build Top section events from other sections so the badge is never 0.
    if len(digest['sections'])>=2:
      seen=set(); top_events=[]
      for sec in digest['sections'][1:]:
        for e in (sec.get('events') or [])[:2]:
          u=e.get('url','')
          if not u or u in seen: continue
          seen.add(u); top_events.append(e)
          if len(top_events)>=10: break
        if len(top_events)>=10: break
      digest['sections'][0]['events']=top_events

    # 2) Structured, scannable Top brief (HTML + plain text)
    basis = digest['sections'][1:] if len(digest['sections'])>=2 else digest['sections']
    tb_txt, tb_html, _ = today_brief_struct(basis, args.model)
    top = digest['sections'][0]
    top['brief_zh']=tb_txt; top['brief_global']=tb_txt; top['brief_cn']=tb_txt
    top['brief_html']=tb_html; top['brief_html_global']=tb_html; top['brief_html_cn']=tb_html

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  render_html(args.template,digest,args.out)
  print('OK:',args.out,'date=',date)
  if not llm_available(): print('NOTE: 未配置 OPENAI_API_KEY：中文翻译/简报为占位。')

if __name__=='__main__': main()
