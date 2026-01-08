# Ben's Daily Brief v2.8

苹果风格新闻简报系统，支持历史回溯和移动端访问。

## v2.8 更新 - 重大修复

| 功能 | 状态 |
|------|------|
| **JSON截断修复** | ✅ 智能修复被截断的JSON |
| **输出长度优化** | ✅ 减少数据量避免截断 |
| **添加按钮样式** | ✅ 蓝色高亮 |
| **表单位置优化** | ✅ 新增时显示在按钮下方 |

## 本次核心修复

### JSON截断问题（根本原因）
之前日志显示 "non-JSON" 错误，实际原因是：
1. LLM返回的JSON太长（9000-11000字符）
2. 被 `max_tokens` 截断后JSON不完整
3. 简单的括号补全无法修复复杂截断

### 解决方案

1. **智能JSON修复** (`repair_truncated_json`)
   - 在events数组中找到最后一个完整的对象
   - 截断不完整的部分
   - 正确闭合JSON结构

2. **减少输出长度**
   - 发送给LLM的新闻数量：30条 → 20条
   - 要求生成的events：15条 → 10条
   - 标题/摘要长度限制更严格
   - Prompt明确要求"不要markdown"

3. **日志改进**
   - 成功修复时显示 `[digest] JSON repaired successfully`

## 预期效果

部署后：
- "non-JSON" 错误应大幅减少
- 即使JSON被截断，也能自动修复
- 新闻翻译成功率显著提升

## 部署

```bash
# 完全替换所有文件
unzip ben-daily-digest-v2.zip
cp -r ben-daily-digest-v2/* your-repo/
cd your-repo
git add -A && git commit -m "v2.8: fix JSON truncation" && git push
```

## 其他功能

- **历史回溯**：点击日期查看历史简报
- **移动端导航**：📰新闻 / ☰导航 / 📊指标 / ⚙设置
- **可视化配置**：设置 → 类目管理 / 数据源

---
版本: v2.8 | 日期: 2026-01-08
