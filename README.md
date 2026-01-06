# Ben Daily Digest v15 - 完整修复版

## 修复内容

### 1. ✅ 设置面板显示优化
- **问题**: 设置只显示JSON代码
- **修复**: 重新设计设置面板，包含5个选项卡:
  - 通用: 基本信息、深色模式
  - 算法调优: LLM模型、每板块条目数、原始抓取限制
  - 显示: 卡片布局、反馈按钮、来源链接开关
  - 数据源: FRED、Google News状态
  - 配置: 完整配置导出

### 2. ✅ 字体一致性
- **问题**: 框架字体不一致
- **修复**: 统一使用 `--font-sans` CSS变量，包含:
  - -apple-system
  - SF Pro Display/Text
  - PingFang SC (中文)
  - Microsoft YaHei
  - Noto Sans CJK SC

### 3. ✅ 算法调优选项
- **问题**: 设置缺少算法调优内容
- **修复**: 新增算法调优面板，显示:
  - LLM模型 (gpt-4o-mini)
  - items_per_section (每板块条目数)
  - limit_raw (原始抓取限制)
  - china_queries_enabled (中国查询增强)
  - 调优建议说明

### 4. ✅ 中间栏卡片化布局
- **问题**: 内容没有板块化、方格化
- **修复**: 
  - 2列网格布局 (cards-grid)
  - 卡片悬停效果
  - 中国相关卡片左侧红色边框标识

### 5. ✅ 反馈按钮
- **问题**: 没有给算法反馈的功能
- **修复**: 
  - 每个事件卡底部添加 👍/👎 按钮
  - 反馈数据存储在 localStorage
  - 右侧栏显示反馈统计
  - 支持导出/清空反馈

### 6. ✅ 板块总结与决策建议
- **问题**: 每个板块顶部没有总结性文字和决策建议
- **修复**:
  - Python脚本增强LLM prompt，要求生成:
    - 【今日主线】
    - 【关键变化】
    - 【决策建议】
  - 模板用渐变背景高亮显示摘要区

### 7. ✅ 中文翻译
- **问题**: 简要内容和标题没有中文翻译
- **修复**:
  - LLM prompt明确要求所有输出为中文
  - 英文新闻强制翻译
  - brief_zh 300-500字中文摘要

### 8. ✅ 中国相关新闻
- **问题**: 每次生成都没有中国相关新闻
- **修复**:
  - 新增 `CHINA_QUERIES` 字典，为每个板块添加专属中国搜索词
  - 使用 zh-CN 语言参数查询 Google News
  - 增强中国关键词检测 (80+ 关键词)
  - 中国相关新闻标记为 region="cn"
  - 左侧导航显示中国新闻数量

### 9. ✅ KPI颜色匹配
- **问题**: 右侧栏数据颜色没有按趋势线匹配
- **修复**:
  - KPI值颜色与趋势线颜色一致
  - 上涨: #16A34A (绿色)
  - 下跌: #DC2626 (红色)
  - 持平: #64748B (灰色)

## 文件清单

```
ben-daily-digest-FIXED/
├── .github/
│   └── workflows/
│       └── generate.yml      # GitHub Actions 配置
├── news_digest_generator_v15.py  # Python脚本 (871行)
├── daily_digest_template_v15_apple.html  # HTML模板 (1064行)
├── digest_config_v15.json    # 配置文件
├── requirements.txt          # Python依赖
├── CNAME                     # GitHub Pages域名
└── .gitignore
```

## 部署步骤

1. **下载修复包**
   - 下载 `ben-daily-digest-FIXED.zip`

2. **替换GitHub仓库**
   ```bash
   # 克隆仓库
   git clone https://github.com/your-username/ben-daily-digest.git
   cd ben-daily-digest
   
   # 删除旧文件
   rm -rf *
   
   # 解压修复包
   unzip ben-daily-digest-FIXED.zip
   mv ben-daily-digest-FIXED/* .
   rm -rf ben-daily-digest-FIXED
   
   # 提交
   git add -A
   git commit -m "v15 complete fix: UI, China news, settings, feedback"
   git push
   ```

3. **手动触发 GitHub Actions**
   - 进入仓库 → Actions → generate.yml → Run workflow

4. **验证**
   - 检查 Actions 日志，确认:
     - `kpi done: 9` 所有KPI有值
     - `gnews-cn` 中国新闻查询成功
     - `llm pack success` LLM摘要成功
   - 访问页面，确认:
     - 设置面板显示完整UI
     - 中间栏卡片布局正常
     - 有中国相关新闻 (红色边框)
     - KPI颜色与趋势线匹配

## 技术改进

### JSON解析增强
```python
def fix_json_string(s: str) -> str:
    # 修复缺少逗号: "value"\n"key": → "value","key":
    s = re.sub(r'"\s*\n\s*"([^"]+)":', r'","\1":', s)
    # 修复数字后缺少逗号
    s = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', s)
    # ...更多修复
```

### 中国新闻查询
```python
CHINA_QUERIES = {
    "macro": ["China PBOC interest rate", "人民币 汇率 央行", ...],
    "metals": ["China copper aluminum demand", "中国 有色金属 进口", ...],
    # 每个板块都有专属查询
}

# 使用中文Google News
items = fetch_google_news_items(q, hl="zh-CN", gl="CN", ceid="CN:zh-Hans")
```

### LLM Prompt增强
```python
sys_prompt = """
- brief_zh必须包含"决策建议"部分
- 优先选择中国相关的新闻(is_china_related=true)
- 所有标题和摘要必须是中文
"""
```

## 注意事项

1. **API限制**: Google News RSS 可能有访问频率限制，建议合理设置抓取数量
2. **LLM成本**: 使用 gpt-4o-mini 降低成本，如需更高质量可改用 gpt-4o
3. **反馈数据**: 存储在浏览器 localStorage，清除浏览器数据会丢失

---

修复版本: v15-FIXED  
日期: 2026-01-06
