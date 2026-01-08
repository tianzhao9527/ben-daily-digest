# Ben Daily Digest v3.0

智能新闻简报生成器 - 金属贸易/地缘政治/AI算力专注版

## v3.0 更新日志

### 🔧 本次修复

**1. 配置文件优先**
- 配置文件中的 `items_per_section` 和 `limit_raw` 现在会被正确读取
- 命令行参数仅作为覆盖选项

**2. 时效性过滤优化**
- 默认放宽到14天（原7天）
- 新闻不足时自动放宽到30天
- 确保每个板块有足够内容

**3. arXiv抓取增强**
- 增加重试次数（4次）和间隔（2秒）
- 添加备用逻辑：失败时使用Google News搜索AI研究相关新闻
- 使用HTTPS协议

**4. 关键词云优化**
- 字体大小统一：10-12px（原10-14px）
- 分3个等级显示
- 更精致的pill样式

### 📊 KPI增强
- **MA5/MA20均线** - 橙色MA5，紫色MA20
- **支撑/阻力位** - 基于20日高低点
- **增强迷你图** - 显示多条均线

### 🎯 市场情绪仪表盘（Apple风格）
- 渐变圆弧进度条
- 精致指针动画
- 看多/中性/看空圆形徽章

### 📅 月日历选择器
- 简约月历视图
- 有数据的日期带蓝点标记
- 左右翻月导航

### 📰 新闻类目（11个）

| ID | 名称 | 搜索词数 |
|----|------|----------|
| macro | 宏观/市场 | 5 |
| sanctions | 地缘政治/制裁 | 5 |
| ai | AI/大模型 | 6 |
| compute | AI计算基础设施 | 5 |
| ev | 新能源汽车 | 6 |
| metals | 大宗/金属 | 5 |
| carbon | 碳/CBAM | 5 |
| sea | 东南亚供应链 | 5 |
| space | 宇宙探索 | 5 |
| frontier | 前沿科技 | 5 |
| arxiv | 研究雷达 | arXiv API |

### ⚙️ 默认配置

```json
{
  "items_per_section": 10,
  "limit_raw": 30
}
```

## 已知问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 某板块新闻=0 | 7天内无相关新闻 | 已放宽到14-30天 |
| arXiv=0 | API限流429 | 添加备用Google News |
| DeepSeek失败 | 网络问题 | 自动切换Qwen |
| 配置不生效 | 代码读取默认值 | 已修复配置优先 |

## GitHub Secrets

| Secret | 必需 | 说明 |
|--------|------|------|
| DEEPSEEK_API_KEY | ✓ | DeepSeek API |
| QWEN_API_KEY | 可选 | 通义千问备用 |
| OPENAI_API_KEY | 可选 | OpenAI备用 |
| GNEWS_API_KEY | ✓ | GNews新闻API |
| FRED_API_KEY | ✓ | FRED经济数据 |

## 升级指南

1. **完整替换所有文件**（包括digest_config_v15.json）
2. 推送到GitHub
3. 手动触发一次Action或等待定时运行
4. 检查日志确认 `limit_raw=30 items=10`

## 许可

MIT License
