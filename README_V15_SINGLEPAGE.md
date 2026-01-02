# V18 单页静态简报（Patch）

## 你会得到什么
- `index.html`：完全静态（打开页面不再抓取）
- 单页三视图：全部 / 中国相关 / 全球（不含中国）——全部预渲染
- “Top 今日要点” + 每个栏目都有 300–500 字栏目简报 + 结论优先事件卡
- 右侧“决策面板”：KPI（含绿/红趋势）+ 阈值触发提示

## 文件
- news_digest_generator_v18_singlepage.py
- daily_digest_template_v18_singlepage.html
- digest_config_v18.json
- .github/workflows/generate.yml（见下方建议内容）

## GitHub Actions 定时（北京时间 06:00）
GitHub schedule 用 UTC，所以 06:00(BJT) = 22:00(UTC)。
cron 用：`0 22 * * *`

## 必要的 Secret
仓库 Settings → Secrets and variables → Actions → New repository secret：
- `OPENAI_API_KEY`

（不配也能跑，但中文翻译/要点会弱化）

## 最小自检
Actions → Run workflow：
- 成功后仓库根目录出现 `index.html`
- Pages 绑定域名后打开能看到内容
