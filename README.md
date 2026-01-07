# Ben's Daily Brief v2.5

苹果风格新闻简报系统，支持历史回溯和移动端访问。

## v2.5 更新

| 功能 | 状态 |
|------|------|
| **历史简报回溯** | ✅ 点击日期查看历史 |
| **移动端底部导航** | ✅ 新闻/导航/指标/设置 |
| **配置可视化编辑** | ✅ 事件委托修复 |
| **自动存档** | ✅ 保留90天历史 |

## 历史简报功能

每天生成的简报会自动存档到 `archive/` 目录：

```
archive/
├── index.json           # 日期索引
├── digest_2026-01-07.json
├── digest_2026-01-06.json
└── ...
```

**使用方式**：点击页面顶部的日期，下拉菜单显示所有可用日期，选择即可查看历史简报。

## 移动端使用

在手机上访问时，底部显示4个导航按钮：

| 按钮 | 功能 |
|------|------|
| 📰 新闻 | 主内容区 |
| ☰ 导航 | 左侧分类导航（可关闭） |
| 📊 指标 | 右侧KPI数据（可关闭） |
| ⚙ 设置 | 配置面板 |

点击左上角 × 按钮关闭导航/指标面板。

## 可视化配置编辑

1. 打开设置 → 类目管理
2. 点击 ✎ 编辑 或 + 添加
3. 填写ID、名称、关键词
4. 保存后点击"导出类目JSON"或"下载配置文件"
5. 将配置文件上传到GitHub替换原文件

## 多LLM API 配置

| 优先级 | 提供商 | Secret名称 |
|--------|--------|------------|
| 1 | DeepSeek | `DEEPSEEK_API_KEY` |
| 2 | 通义千问 | `QWEN_API_KEY` |
| 3 | OpenAI | `OPENAI_API_KEY` |

## 文件结构

```
├── index.html                  # 当天简报
├── archive/                    # 历史存档
│   ├── index.json             # 日期索引
│   └── digest_YYYY-MM-DD.json # 每日数据
├── news_digest_generator_v15.py
├── daily_digest_template_v15_apple.html
├── digest_config_v15.json
└── .github/workflows/generate.yml
```

## 部署

```bash
unzip ben-daily-digest-v2.zip
mv ben-daily-digest-v2/* your-repo/
git add -A && git commit -m "v2.5" && git push
```

---
版本: v2.5 | 日期: 2026-01-08
