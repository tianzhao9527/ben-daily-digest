# Ben's Daily Brief v2.1

干净的苹果风格新闻简报系统，支持多LLM自动切换确保翻译成功。

## 更新内容

| 功能 | 状态 |
|------|------|
| 多LLM支持 | ✅ DeepSeek → Qwen → OpenAI 自动切换 |
| 新闻日期显示 | ✅ 每个卡片显示发布日期 |
| 类目管理 | ✅ 设置中可查看/添加类目 |
| 卡片布局优化 | ✅ 两列卡片，更好的可读性 |
| 时效性优化 | ✅ 7天过滤，24/48h加权 |

## 多LLM API 配置

系统支持三个LLM提供商，按优先级自动切换：

| 优先级 | 提供商 | Secret名称 | 获取地址 |
|--------|--------|------------|----------|
| 1 | DeepSeek | `DEEPSEEK_API_KEY` | https://platform.deepseek.com/ |
| 2 | 通义千问 | `QWEN_API_KEY` | https://dashscope.console.aliyun.com/ |
| 3 | OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/ |

**设置 GitHub Secrets**:
1. 仓库 **Settings** → **Secrets and variables** → **Actions**
2. 添加至少一个 API Key

**提示**: 配置多个API Key可确保翻译不会因单一服务故障而失败。

## 调度时间

- **北京时间每天上午 7:00** 自动运行
- 也可以随时手动触发

## 类目管理

在设置面板的"类目管理"选项卡中查看当前类目。

**添加新类目**: 编辑 `digest_config_v15.json`，在 `sections` 数组中添加：

```json
{
  "id": "new_section",
  "name": "新板块名称",
  "gnews_queries": ["关键词1", "关键词2"]
}
```

提交后 **T+1** (次日) 生效。

## KPI 指标

| 分类 | 指标 |
|------|------|
| 美股指数 | 标普500, 纳斯达克, 道琼斯, VIX |
| 宏观指标 | 美债10Y/2Y, USD/CNY, 美元指数 |
| 大宗商品 | WTI原油 |
| 金属价格 | 铜, 铝, 锌, 镍 |

## 部署

```bash
unzip ben-daily-digest-v2.zip
mv ben-daily-digest-v2/* your-repo/
git add -A && git commit -m "v2.1" && git push
```

## 验证

运行后检查日志：
- `[digest] trying LLM: DeepSeek` - 尝试的LLM
- `[digest] llm pack success (xxx) by DeepSeek` - 翻译成功
- 如果失败会自动切换到下一个LLM

---
版本: v2.1 | 日期: 2026-01-07 | LLM: DeepSeek/Qwen/OpenAI
