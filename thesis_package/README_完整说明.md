# OSrCIR 毕业设计论文 — 完整资料包说明

> 打包日期: 2026-04-16
> 项目: 基于视觉代理与描述融合的零样本组合式图像检索改进
> 作者: 杨昊明 | 同济大学 电子与信息工程学院 计算机科学与技术

---

## 文件夹结构总览

```
thesis_package/
├── README_完整说明.md          ← 你正在看的这个文件
├── 论文版本/                   ← 所有版本的论文稿
│   ├── v1_同济格式_markdown初稿.md
│   ├── v2_通用本科模板草稿.md
│   ├── v3_通用模板骨架.md
│   ├── v4_通用LaTeX模板.tex
│   ├── v5_同济格式_最终LaTeX.tex
│   └── v5_同济格式_最终.pdf
├── 图表及代码/                 ← 所有论文图表 + 生成脚本
│   ├── generate_all_figures.py
│   ├── fig_pipeline.pdf / .png
│   ├── fig_main_results.pdf / .png
│   ├── fig_prompt_evolution.pdf / .png
│   ├── fig_heatmap.pdf / .png
│   ├── fig_fashioniq.pdf / .png
│   ├── fig_ablation.pdf / .png
│   └── fig_relative.pdf / .png
└── 实验数据/                   ← 原始实验结果 JSON/日志/报告
    ├── FINAL_RESULTS.md
    ├── GRID_SEARCH_REPORT.md
    ├── EXPERIMENT_LOG.md
    ├── PROGRESS_REPORT.md
    ├── eval_summary.json
    ├── eval_summary_gpu.json
    ├── genecis_eval_summary.json
    ├── genecis_grid_search_full.json
    ├── genecis_grid_search_full_detail.json
    └── grid_search_quickgelu.log
```

---

## 一、论文版本说明

| 文件 | 说明 |
|------|------|
| `v1_同济格式_markdown初稿.md` | 按同济大学撰写规范写的 Markdown 初稿，包含封面、中英文摘要、5 章正文、参考文献、谢辞。格式标注用 HTML 注释 |
| `v2_通用本科模板草稿.md` | 早期通用本科模板草稿，章节编号为"第X章"形式 |
| `v3_通用模板骨架.md` | 最简骨架模板，仅有章节标题和占位内容 |
| `v4_通用LaTeX模板.tex` | 早期 LaTeX 版本（pandoc 风格，章节用"第X章"） |
| **`v5_同济格式_最终LaTeX.tex`** | **当前最终版** — 完整 LaTeX 源文件，符合同济格式，包含所有图表嵌入 |
| **`v5_同济格式_最终.pdf`** | **当前最终版 PDF** — 26 页，可直接阅读/打印 |

### 版本演进路线

```
v3 (骨架) → v2 (通用草稿) → v1 (同济 Markdown) → v4 (通用 LaTeX) → v5 (同济 LaTeX + 图表，最终版)
```

---

## 二、图表详细说明

### 图 1: `fig_pipeline` — 三路融合方法总体流程图

**作用**: 论文第 3 章的核心配图，展示整个方法从输入到输出的完整流程。

**内容解读**:
- **左侧 (Stage 1)**: 参考图像 + 修改文本 → MLLM (Qwen-VL-Max) → 初始描述 D₁ → 文生图模型 (MiniMax image-01) → 代理图像
- **中间 (Stage 2)**: 参考图像 + 代理图像 + 修改文本 → MLLM (V7 反幻觉 prompt) → 精炼描述 D₂
- **右侧 (Retrieval)**: CLIP 编码三路信号，按公式 `score = α·sim(f_text, gallery) + (1-α)·sim(CLIP(proxy), gallery)` 计算最终排序
- 底部标注两个关键超参数: β=0.7（文本融合权重）和 α=0.9（三路融合权重）

**设计思路**: 用 matplotlib 手动绘制方框 + 箭头，替代 Mermaid/TikZ，确保跨平台兼容且可直接编译进 LaTeX。

---

### 图 2: `fig_main_results` — 9 个数据集主指标对比

**作用**: 论文第 4 章的总览图，一眼看出所有数据集的提升情况。

**内容解读**:
- 蓝色柱 = Baseline (OSrCIR 复现)
- 红色柱 = 三路融合方法 (Ours)
- 每个红色柱顶部标注绝对提升值（如 +3.5, +5.1 等）
- 虚线分隔: 左侧 5 个数据集用默认参数 (β=0.7, α=0.9)，右侧 4 个 GeneCIS 子集用 task-adaptive 参数
- 主指标: FashionIQ 用 R@10，CIRCO 用 mAP@10，CIRR 和 GeneCIS 用 R@1

**关键数据**:

| 数据集 | Baseline | 三路融合 | 绝对提升 | 相对提升 |
|--------|----------|---------|---------|---------|
| FIQ dress | 15.80 | 19.29 | +3.49 | +22.1% |
| FIQ shirt | 26.00 | 27.35 | +1.35 | +5.2% |
| FIQ toptee | 23.09 | 27.35 | +4.26 | +18.4% |
| CIRCO | 16.21 | 21.26 | +5.05 | +31.2% |
| CIRR | 22.96 | 25.90 | +2.94 | +12.8% |
| GeneCIS ch_obj | 13.83 | 25.51 | +11.68 | +84.5% |
| GeneCIS fo_obj | 16.02 | 23.62 | +7.60 | +47.4% |
| GeneCIS ch_attr | 12.65 | 21.79 | +9.14 | +72.3% |
| GeneCIS fo_attr | 18.82 | 27.83 | +9.01 | +47.9% |

**结论**: 35/35 指标全部提升，0 退化。

---

### 图 3: `fig_prompt_evolution` — Prompt 迭代对比图

**作用**: 论文第 3.3.3 节配图，展示 prompt 工程的迭代历程和关键突破点。

**内容解读** (数据来自 CIRR 200 样本测试):

| Prompt 版本 | CIRR R@10 | 变化 | 颜色 | 说明 |
|------------|-----------|------|------|------|
| Baseline (D₁ only) | 67.0 | 基准 | 灰色 | 不做精炼，仅用第一轮描述 |
| Original prompt | 64.5 | -2.5 | 橙色 | 直接对比原图和代理图写描述，幻觉严重 |
| V5 (CoT) | 65.5 | -1.5 | 橙色 | 加入思维链结构，略有改善但幻觉未解决 |
| V6 (full context) | 49.5 | **-17.5** | 红色 | 把第一轮完整输出都传给 MLLM，信息过载导致灾难性崩溃 |
| V7 (anti-hallucination) | 73.5 | **+6.5** | 绿色 | 关键突破: "代理图只做诊断，不做描述来源" |
| V7 + Ensemble | 69.0 | +2.0 | 蓝色 | 融合 D₁/D₂ 后牺牲极端提升换取全局稳定 |

**核心发现**: V6 的灾难性退化证明了"信息过多反而有害"；V7 的突破证明了"限制信息来源比增加信息更重要"。

---

### 图 4: `fig_heatmap` — GeneCIS α/β 网格搜索热力图

**作用**: 论文第 4.3.2 节配图，展示参数敏感性和各子集的差异化偏好。

**内容解读**:
- 3 个子图: change_object, focus_object, focus_attribute（change_attribute 因 gallery 缓存问题无完整热力图数据，在表格中补充）
- X 轴 = α (0.70~1.00): 文本 vs 代理图权重。α 越大越依赖文本。
- Y 轴 = β (0.3~1.0): D₁ vs D₂ 权重。β 越大越依赖原始描述 D₁。
- 颜色: 绿色 = 相比 Baseline 提升, 红色 = 下降
- ★ 标记各子集最优参数组合

**关键发现**:
- **change_object**: 最优 β=0.30, α=0.95 — 几乎只用 D₂（D₂ 占 70%），几乎不用代理图（仅 5%）。说明 V7 精炼描述对 "物体变化" 任务价值极大。
- **focus_object**: 最优 β=1.00, α=0.90 — 完全不用 D₂！因为 V7 的"描述不超过修改文本长度"限制把 focus_object 的 D₂ 压缩到 1-3 个词，反而是噪声。
- **focus_attribute**: 最优 β=0.60, α=0.85 — D₁ 和 D₂ 各占约 60%/40%，代理图权重 15%，三路信号都在发挥作用。

**设计思路**: 数据来自 `grid_search_quickgelu.log`，56 种参数组合 × 3 个子集共 168 个数据点。

---

### 图 5: `fig_fashioniq` — FashionIQ 三子集详细结果

**作用**: 论文第 4.2.2 节配图，展示 FashionIQ 三个服饰子集在 R@1/5/10/50 上的全面提升。

**内容解读**:
- 3 个子图分别对应 dress, shirt, toptee
- 每个子图包含 4 个指标 (R@1, R@5, R@10, R@50) 的 Baseline vs 三路融合对比
- dress 提升最大 (R@50 +5.42)，shirt 提升最保守 (R@10 +1.35)
- 12/12 指标全部正向

---

### 图 6: `fig_ablation` — 各模块消融对比

**作用**: 论文第 4.3.3 节配图，展示逐步添加各模块的效果变化。

**内容解读** (200 样本):
- 5 种配置: Baseline → +Proxy(后融合) → +Refinement(V7, 仅D₂) → +Ensemble(D₁+D₂) → Three-Way(完整)
- 左图 CIRR R@10: V7 refinement 带来最大提升 (67→73.5)，但 Ensemble 回调一部分 (73.5→69)，最终 Three-Way=69
- 右图 FIQ dress R@10: Proxy 有效 (15.5→17)，但 V7 refinement 反而退化 (→13)，Ensemble 修复退化 (→18)，Three-Way=19

**核心洞察**: 单模块最优配置在不同数据集上截然相反——CIRR 上 V7 是主力，FIQ 上 Ensemble 是关键。三路融合的价值在于统一框架下兼顾两端。

---

### 图 7: `fig_relative` — 相对提升百分比横向对比

**作用**: 论文第 4.2.1 节配图，直观展示哪些数据集受益最大。

**内容解读**:
- 横向条形图，按数据集排列
- GeneCIS 提升最为惊人 (ch_obj +84.5%, ch_attr +72.3%)
- CIRCO 也很显著 (+31.2%)
- shirt 提升最小 (+5.2%)，但仍然是正向的

---

## 三、研究思路完整梳理

### 3.1 问题出发点

OSrCIR 用 MLLM 生成一段目标描述 → CLIP 编码 → 检索。问题是：
1. **只有一次机会**: 如果描述错了，没有纠错机制
2. **只走文本路径**: 缺乏图像空间的互补信号
3. **结果脆弱**: CLIP 对文本表述敏感，单一描述的随机波动影响大

### 3.2 三层创新

```
创新1: Visual Proxy（视觉代理）
  D₁ → 文生图 → 代理图
  作用: 提供图像空间辅助信号 + 为精炼提供视觉参照

创新2: Anti-Hallucination CoT（反幻觉提示词）
  参考图 + 代理图 + 修改文本 → MLLM (V7 prompt) → D₂
  核心: 代理图只做诊断工具，不做描述来源

创新3: Description Ensemble + Three-Way Fusion（描述融合 + 三路融合）
  f_text = normalize(β·CLIP(D₁) + (1-β)·CLIP(D₂))
  score = α·sim(f_text, gallery) + (1-α)·sim(CLIP(proxy), gallery)
  作用: 兜底机制——即使 D₂ 退化，D₁ 仍占主导
```

### 3.3 迭代过程（关键教训）

| 版本 | 做了什么 | 结果 | 教训 |
|------|---------|------|------|
| Original prompt | 直接让 MLLM 对比原图和代理图 | CIRR -2.5 | 代理图的幻觉细节被直接复制 |
| V5 (CoT) | 加思维链结构 | CIRR -1.5 | 结构化推理不够，幻觉路径没被切断 |
| V6 (full ctx) | 把 D₁ 全文也传给 MLLM | CIRR **-17.5** | 灾难：信息过载导致 MLLM 过度修改 D₁ |
| **V7** | "代理图只做诊断" + 强制短描述 | CIRR **+6.5**, 但 shirt -4 | 突破！但过于激进导致部分子集退化 |
| V7 + Ensemble | 在 CLIP 空间融合 D₁/D₂ | 全部正向 | 融合是"万金油"——牺牲极端提升换稳定 |
| V7 + 三路融合 | 再加代理图后融合 | 35/35 提升 | 最终方案 |
| GeneCIS 专用 prompt | 去掉"不超过修改文本长度"限制 | 大幅提升 | 短文本任务需要专门优化 |

### 3.4 关键设计决策

**为什么 β=0.7（D₁ 占 70%）？**
- D₁ 是 MLLM 直接理解的结果，通常更完整
- D₂ 是精炼后的结果，更精准但可能遗漏信息
- β=0.7 = 以 D₁ 兜底 + D₂ 修正

**为什么 α=0.9（文本占 90%）？**
- 代理图是 AI 生成的，必然包含幻觉
- 10% 的权重足以提供辅助信号，不会主导检索方向

**为什么 GeneCIS 要用不同参数？**
- GeneCIS 修改文本只有 1-2 个词（如 "bus"、"color"），V7 的"描述不超过修改文本"限制导致 D₂ 退化为 2-3 个词
- 每个查询只有 ~14 张候选图，代理图的场景偏好影响更大
- 必须用专用 prompt + task-adaptive α/β

---

## 四、实验数据文件说明

| 文件 | 说明 |
|------|------|
| `FINAL_RESULTS.md` | **最核心**: 全量 + 200样本完整数值结果，35 项指标汇总 |
| `GRID_SEARCH_REPORT.md` | GeneCIS α/β 网格搜索完整报告，含流程图、参数含义、各子集结果、专用 prompt 实验 |
| `EXPERIMENT_LOG.md` | 实验日志（缓存资产、花费、环境信息） |
| `PROGRESS_REPORT.md` | 向导师汇报的完整文档 |
| `eval_summary.json` | FIQ/CIRCO/CIRR 全量 JSON 结果（Windows CLIP 编码） |
| `eval_summary_gpu.json` | GPU 版评估结果 |
| `genecis_eval_summary.json` | GeneCIS 4 子集全量 JSON 结果 |
| `genecis_grid_search_full.json` | 网格搜索最优参数 JSON |
| `genecis_grid_search_full_detail.json` | 网格搜索 56×3 全部数据点 |
| `grid_search_quickgelu.log` | 网格搜索运行日志（含完整排序表） |

---

## 五、图表生成方法

所有 7 张图由 `图表及代码/generate_all_figures.py` 一键生成:

```bash
cd 图表及代码
python3 generate_all_figures.py
```

依赖: `matplotlib`, `numpy`（`pip install matplotlib numpy`）

输出: 每张图同时生成 `.pdf`（LaTeX 用）和 `.png`（预览用）。

---

## 六、如何编译论文 PDF

```bash
cd 论文版本
# 需要 texlive + ctex 宏包 + xelatex
# 图片需要在上级目录的 figures/ 中，或者把 fig_*.pdf 复制到同目录
xelatex thesis.tex && xelatex thesis.tex
```

或者直接使用已编译好的 `v5_同济格式_最终.pdf`。

---

## 七、花费统计

| 项目 | 花费 |
|------|------|
| MLLM API (Qwen-VL-Max, 阿里云 DashScope) | ~90 元 |
| 文生图 API (MiniMax image-01) | ~50 元 |
| **合计** | **~140 元** |
