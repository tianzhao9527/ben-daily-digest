# Ben 的每日资讯简报（GitHub Pages 部署版）

这个仓库用于**定时生成**一个静态 `index.html`（无客户端抓取），并通过 **GitHub Pages** 对外提供访问。

## 你会得到什么
- 访问 `https://<你的用户名>.github.io/<仓库名>/` 就能看到最新简报
- GitHub Actions 会在 **北京时间 08:00 / 16:00 / 00:00**（每 8 小时）自动更新 `index.html`

## 必需文件
- `news_digest_generator_v15.py`：生成器
- `daily_digest_template_v15_apple.html`：页面模板
- `digest_config_v15.json`：栏目与 KPI 配置
- `.github/workflows/generate.yml`：定时任务（Actions）

## 可选：启用更强的中文总结
生成器支持使用 OpenAI 生成更好的中文栏目简报与“今日要点”。

在 GitHub 仓库中设置 `OPENAI_API_KEY`：
Settings → Secrets and variables → Actions → New repository secret

Name: `OPENAI_API_KEY`
Value: 你的 key

## 修改栏目 / KPI（可视化）
打开站点页面 → 右下角【设置】→【扩展】：
- 新增/编辑类目与 KPI
- 点击【导出配置】得到 `digest_config_override.json`
- 把它覆盖仓库里的 `digest_config_v15.json`（提交后下次生成生效）
