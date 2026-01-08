# Ben Daily Digest v3.0

智能新闻简报生成器 - 金属贸易/地缘政治/AI算力专注版

## v3.0 新功能

### 1. 新闻去重
- Jaccard相似度算法（阈值0.5）
- 相似新闻自动合并，保留所有来源链接
- 减少LLM token消耗

### 2. 情感分析
- 每个板块独立情感判断：看多/中性/看空
- 情感分数 0-1（0看空，1看多）
- 在板块标题旁显示

### 3. 关键词提取
- LLM自动提取3-5个关键词
- 以标签形式显示在板块头部

### 4. GitHub配置同步
- 在「设置→同步」tab中配置
- 支持完整URL或 owner/repo 格式
- 直接通过API推送配置更改
- T+1自动生效

## 文件结构

```
├── news_digest_generator_v15.py    # 主生成脚本
├── daily_digest_template_v15_apple.html  # HTML模板
├── digest_config_v15.json          # 配置文件
├── .github/workflows/generate.yml  # GitHub Actions
├── archive/                        # 历史存档
└── requirements.txt
```

## 配置文件说明

```json
{
  "items_per_section": 15,
  "limit_raw": 25,
  "importance_weights": {...},
  "sections": [...],
  "kpis": [...]
}
```

## GitHub Secrets 配置

| Secret | 必需 | 说明 |
|--------|------|------|
| DEEPSEEK_API_KEY | ✓ | DeepSeek API密钥 |
| QWEN_API_KEY | 可选 | 通义千问备用 |
| OPENAI_API_KEY | 可选 | OpenAI备用 |
| GNEWS_API_KEY | ✓ | GNews新闻API |
| FRED_API_KEY | ✓ | FRED经济数据API |

## 本地运行

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=xxx
export GNEWS_API_KEY=xxx
export FRED_API_KEY=xxx
python news_digest_generator_v15.py
```

## 设置面板功能

- **通用**: 基本信息、主题切换
- **类目管理**: 可视化编辑新闻板块
- **重要性系数**: 调整新闻权重
- **算法**: 查看LLM配置
- **数据源**: 编辑KPI指标
- **配置**: 导出完整配置JSON
- **同步**: GitHub配置同步（新增）

## 升级指南

1. 解压替换所有文件
2. 推送到GitHub
3. 无需修改Secrets
4. 如需使用GitHub同步功能，在设置中配置Token

## 许可

MIT License
