# Ben Daily Digest (v15)

这是一个 **静态** 的每日资讯简报生成器：GitHub Actions 定时运行，生成 `index.html` 并提交回仓库；GitHub Pages 直接展示 `index.html`。

## 文件说明
- `news_digest_generator_v15.py` 生成器（抓取 + 聚类 + LLM 总结 + 输出 HTML）
- `digest_config_v15.json` 配置（栏目、查询、KPI、阈值）
- `daily_digest_template_v15_apple.html` 页面模板（Apple 风格单页）
- `.github/workflows/generate.yml` 定时工作流（默认每 8 小时）
- `requirements.txt` Python 依赖

## 你需要配置
在 GitHub 仓库 Settings → Secrets and variables → Actions：
- 新建 `OPENAI_API_KEY`（用于中文总结与事件卡）

## GitHub Pages
Settings → Pages：
- Source 选择 **Deploy from a branch**
- Branch 选 `main`，Folder 选 `/ (root)`

## 本地测试
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python news_digest_generator_v15.py \
  --config digest_config_v15.json \
  --template daily_digest_template_v15_apple.html \
  --out index.html

python -m http.server 8000
# 浏览器打开 http://localhost:8000/index.html
```

## 常见问题
- events=0：通常是 RSS/Google News 抓取为空（网络/查询过窄/源异常）。查看 Actions 日志里 `[digest] gnews ... entries=...`
- 卡住很久：脚本对 OpenAI/抓取均有 timeout；若仍卡住，请把 Actions 日志贴出来继续排查。
