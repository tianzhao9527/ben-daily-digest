# Ben's Daily Brief v2.6

苹果风格新闻简报系统，支持历史回溯和移动端访问。

## v2.6 更新

| 功能 | 状态 |
|------|------|
| **JSON解析修复** | ✅ 正确处理markdown代码块 |
| **翻译成功率提升** | ✅ 更健壮的JSON清理 |
| **events数量限制放宽** | ✅ 根据原始数据调整 |
| **历史简报回溯** | ✅ 点击日期查看历史 |
| **移动端底部导航** | ✅ 新闻/导航/指标/设置 |

## 本次修复

### JSON解析问题
LLM经常返回被 \`\`\`json 包裹的JSON，之前的正则无法正确处理。现在使用更健壮的方法：
1. 检测 \`\`\` 开头并移除第一行
2. 移除结尾的 \`\`\`
3. 查找JSON边界并解析

### events数量限制
之前要求至少 `items_per_section // 4` 条events，当原始数据少时会导致重试失败。现在根据实际数据量调整：
```python
min_events = min(len(raw_items), max(1, items_per_section // 4))
```

## 历史简报功能

每天生成的简报会自动存档到 `archive/` 目录。点击页面顶部的日期可查看历史简报。

## 移动端使用

底部显示4个导航按钮：

| 按钮 | 功能 |
|------|------|
| 📰 新闻 | 主内容区 |
| ☰ 导航 | 左侧分类 |
| 📊 指标 | 右侧KPI |
| ⚙ 设置 | 配置面板 |

## 多LLM API 配置

| 优先级 | 提供商 | Secret名称 |
|--------|--------|------------|
| 1 | DeepSeek | `DEEPSEEK_API_KEY` |
| 2 | 通义千问 | `QWEN_API_KEY` |
| 3 | OpenAI | `OPENAI_API_KEY` |

建议同时配置多个API Key以提高成功率。

## 部署

```bash
unzip ben-daily-digest-v2.zip
mv ben-daily-digest-v2/* your-repo/
git add -A && git commit -m "v2.6" && git push
```

---
版本: v2.6 | 日期: 2026-01-08
