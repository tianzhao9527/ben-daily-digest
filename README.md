# Ben's Daily Brief v2

干净的苹果风格新闻简报系统，使用 DeepSeek API 进行翻译和摘要。

## 修复内容

| 问题 | 状态 |
|------|------|
| 去掉所有emoji | ✅ 已移除 |
| 标题改为 Ben's Daily Brief | ✅ 已修改 |
| 设置显示重要性系数 | ✅ 新增"重要性系数"选项卡 |
| 字体大小优化 | ✅ 统一使用 SF Pro 字体系统 |
| 使用 DeepSeek 替代 GPT | ✅ 已切换 |
| 置顶 Top 5 重要新闻 | ✅ 新增决策要点板块 |
| 全球新闻翻译成中文 | ✅ LLM强制翻译 |
| 板块结构化总结 | ✅ 四段式结构输出 |
| 研究雷达(arXiv)无内容 | ✅ 修复arXiv API查询 |
| 新闻时效性 | ✅ 只保留48小时内新闻，24小时内加权 |
| 每日定时抓取 | ✅ 北京时间每天上午7点 |
| 增加美股指数 | ✅ 标普500/纳斯达克/道琼斯/VIX |

## 调度时间

- **北京时间每天上午 7:00** 自动运行
- UTC 时间: 前一天 23:00
- 也可以随时手动触发

## 新增 KPI 指标

### 美股指数
- 标普500 (SP500)
- 纳斯达克 (NASDAQCOM)  
- 道琼斯 (DJIA)
- VIX恐慌指数 (VIXCLS)

### 宏观指标
- 美债10Y/2Y
- USD/CNY 汇率
- 美元指数

### 大宗商品
- WTI原油
- 黄金

### 金属价格
- 铜、铝、锌、镍

## DeepSeek API 配置

### 1. 获取 API Key

1. 访问 [DeepSeek Platform](https://platform.deepseek.com/)
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的 API Key

### 2. 设置 GitHub Secrets

1. 进入你的 GitHub 仓库
2. 点击 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret**
4. 设置:
   - Name: `DEEPSEEK_API_KEY`
   - Secret: 你的 DeepSeek API Key
5. 点击 **Add secret**

## 重要性系数配置

在 `digest_config_v15.json` 中修改：

```json
{
  "importance_weights": {
    "china_related": 1.5,    // 中国相关新闻权重 (建议 1.0-2.0)
    "source_tier1": 1.3,     // 一级来源权重 (Bloomberg, Reuters, FT)
    "source_tier2": 1.1,     // 二级来源权重 (WSJ, SCMP)
    "recency_24h": 1.2,      // 24小时内新闻权重
    "keyword_match": 1.2     // 关键词匹配权重
  }
}
```

数值含义：
- `1.0` = 正常权重
- `1.5` = 提升 50%
- `2.0` = 提升 100%

## 部署步骤

```bash
# 1. 克隆你的仓库
git clone https://github.com/YOUR_USERNAME/ben-daily-digest.git
cd ben-daily-digest

# 2. 删除旧文件
rm -rf *

# 3. 解压新版本
unzip ben-daily-digest-v2.zip
mv ben-daily-digest-v2/* .
rm -rf ben-daily-digest-v2

# 4. 提交推送
git add -A
git commit -m "v2: DeepSeek + Top5 + Clean UI"
git push

# 5. 设置 DEEPSEEK_API_KEY (见上方说明)

# 6. 手动触发 Actions
# GitHub → Actions → Generate Daily Digest → Run workflow
```

## 文件结构

```
├── .github/
│   └── workflows/
│       └── generate.yml      # GitHub Actions 配置
├── news_digest_generator_v15.py  # Python 生成脚本
├── daily_digest_template_v15_apple.html  # HTML 模板
├── digest_config_v15.json    # 配置文件 (重要性系数在这里)
├── requirements.txt          # Python 依赖
└── CNAME                     # GitHub Pages 域名
```

## 新增功能

### Top 5 决策要点

每日自动选出最重要的 5 条新闻，显示在页面顶部，包含：
- 新闻标题
- 决策导向的总结 (200字以内)
- 建议采取的行动

### 板块结构化总结

每个板块的摘要使用四段式结构：
- 【核心动态】今日主要事件
- 【关键数据】重要数据变化
- 【趋势研判】短期走势判断
- 【决策参考】建议采取的行动

### arXiv 研究雷达

自动抓取人工智能、机器学习、大语言模型相关的最新论文，翻译成中文。

## 验证清单

运行后检查 GitHub Actions 日志，确认：
- `[digest] boot v15-DeepSeek` 显示
- `[digest] arxiv arxiv entries=XX` arXiv 抓取成功
- `[digest] llm pack success` LLM 翻译成功
- `[digest] top 5 done` Top 5 生成成功

## 故障排除

### DeepSeek API 调用失败

1. 检查 API Key 是否正确设置
2. 检查 DeepSeek 账户余额
3. 查看 Actions 日志中的具体错误

### arXiv 无内容

1. arXiv API 可能有访问频率限制
2. 检查网络连接
3. 尝试修改 `arxiv_query` 为更简单的查询

### 翻译质量问题

可以尝试在 Python 脚本中调整 DeepSeek 的 `temperature` 参数（当前为 0.3）。

---

版本: v2.0  
日期: 2026-01-07  
LLM: DeepSeek Chat
