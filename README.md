# Ben Daily Digest v3.2

智能新闻简报生成器 - 金属贸易/地缘政治/AI算力专注版

## v3.2 性能优化

### ⚡ 速度提升

| 优化项 | 之前 | 之后 |
|--------|------|------|
| 总运行时间 | ~22分钟 | ~8-10分钟 |
| RSS重试次数 | 3次 | 2次 |
| RSS超时 | 10s+30s | 5s+10s |
| GNews重试 | 3次 | 2次 |
| GNews超时 | 8s+25s | 5s+15s |
| 抓取方式 | 串行 | **并发** |

### 🔄 并发抓取

```python
# RSS源：3个线程并发
ThreadPoolExecutor(max_workers=3)

# Google News：4个线程并发
ThreadPoolExecutor(max_workers=4)
```

### 📰 可用RSS源（经验证）

#### 中国财经
| ID | 名称 | URL |
|----|------|-----|
| `sina_finance` | 新浪财经 | 官方RSS，稳定 |
| `sina_stock` | 新浪股票 | 官方RSS，稳定 |
| `ifeng_finance` | 凤凰财经 | 官方RSS |

#### 全球财经
| ID | 名称 | 状态 |
|----|------|------|
| `bbc_business` | BBC Business | ✅稳定 |
| `cnbc_world` | CNBC World | ✅稳定 |
| `marketwatch` | MarketWatch | ✅稳定 |
| `yahoo_finance` | Yahoo Finance | ✅稳定 |

#### 科技新闻
| ID | 名称 | 状态 |
|----|------|------|
| `techcrunch` | TechCrunch | ✅稳定 |
| `wired` | Wired | ✅稳定 |
| `arstechnica` | Ars Technica | ✅稳定 |
| `theverge` | The Verge | ✅稳定 |
| `engadget` | Engadget | ✅稳定 |

#### 全球新闻
| ID | 名称 | 状态 |
|----|------|------|
| `bbc_world` | BBC World | ✅稳定 |
| `aljazeera` | Al Jazeera | ✅稳定 |
| `npr` | NPR News | ✅稳定 |
| `guardian_world` | The Guardian | ✅稳定 |

#### 商品
| ID | 名称 | 状态 |
|----|------|------|
| `mining` | Mining.com | ✅稳定 |
| `oilprice` | OilPrice | ✅稳定 |

### ❌ 已移除（不可用）

| 源 | 原因 |
|----|------|
| RSSHub (财联社/36氪等) | 403限流 |
| Reuters feeds | DNS解析失败 |
| Kitco | 404 |
| FT | 付费墙 |

### 📊 当前配置

```json
{
  "macro": ["sina_finance", "bbc_business", "marketwatch"],
  "sanctions": ["bbc_world", "aljazeera", "guardian_world"],
  "ai": ["techcrunch", "theverge", "arstechnica"],
  "compute": ["techcrunch", "wired", "engadget"],
  "ev": ["techcrunch", "engadget"],
  "metals": ["mining", "oilprice"],
  "carbon": ["bbc_business", "guardian_world"],
  "sea": ["aljazeera", "bbc_world"],
  "space": ["techcrunch", "arstechnica", "wired"],
  "frontier": ["techcrunch", "wired", "arstechnica"]
}
```

## 其他v3功能

### 📊 KPI增强
- MA5/MA20均线
- 支撑/阻力位

### 🎯 市场情绪
- Apple风格仪表盘
- 看多/中性/看空分布

### 📅 月日历
- 简约月历视图
- 有数据标记

## GitHub Secrets

| Secret | 必需 | 说明 |
|--------|------|------|
| DEEPSEEK_API_KEY | ✓ | DeepSeek API |
| QWEN_API_KEY | 可选 | 通义千问备用 |
| GNEWS_API_KEY | ✓ | GNews API |
| FRED_API_KEY | ✓ | FRED数据 |

## 自建RSSHub

如需财联社等源，建议自建RSSHub：

```bash
docker run -d -p 1200:1200 diygod/rsshub
```

然后修改 `RSS_FEEDS` 中的URL：
```python
"cls_telegraph": {
    "url": "http://localhost:1200/cls/telegraph",
    ...
}
```

## 许可

MIT License
