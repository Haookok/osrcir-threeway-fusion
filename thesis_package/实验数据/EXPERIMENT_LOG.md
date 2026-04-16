# 实验进展日志

> 最后更新: 2026-03-30

---

## 一、已验证完成的核心结果

### 三路融合 (V7 + Description Ensemble β=0.7 α=0.9)

在 Windows RTX 4060 上验证，FashionIQ 3子集 + CIRR，200随机样本(seed=42)。

| 数据集 | 指标 | Baseline | Ensemble | Delta | 状态 |
|--------|:---:|:---:|:---:|:---:|:---:|
| dress | R@10 | 15.5 | **18.0** | **+2.5** | ✅ |
| dress | R@50 | 30.0 | **37.0** | **+7.0** | ✅ |
| shirt | R@10 | 30.0 | **30.5** | **+0.5** | ✅ |
| shirt | R@50 | 45.0 | **47.5** | **+2.5** | ✅ |
| toptee | R@10 | 24.0 | **31.5** | **+7.5** | ✅ |
| toptee | R@50 | 42.0 | **47.0** | **+5.0** | ✅ |
| CIRR | R@10 | 67.0 | **69.0** | **+2.0** | ✅ |
| CIRR | R@50 | 92.0 | **92.5** | **+0.5** | ✅ |

**8/8 指标全部正增长。**

### V7 prompt 单独效果（之前在服务器上验证）

| 数据集 | 指标 | Baseline | V7 A+B最优 | Delta |
|--------|:---:|:---:|:---:|:---:|
| CIRR | R@10 | 67.0 | **73.5** (α=0.8) | **+6.5** |
| dress | R@10 | 15.0 | **20.0** (α=0.8) | **+5.0** |
| toptee | R@10 | 24.0 | **31.5** (α=0.8) | **+7.5** |
| shirt | R@10 | 30.0 | 26.0 (α=0.8) | -4.0 ← Ensemble修复了这个 |
| GeneCIS focus_obj | R@1 | 16.0 | **19.5** (α=0.7) | **+3.5** |
| GeneCIS change_attr | R@1 | 13.0 | **13.5** (α=0.7) | **+0.5** |
| GeneCIS change_obj | R@1 | 10.5 | 9.0 | -1.5 ← 需要Ensemble验证 |
| GeneCIS focus_attr | R@1 | 25.0 | 23.0 | -2.0 ← 需要Ensemble验证 |

---

## 二、待完成的实验

### CIRCO
- v7 精炼缓存：**未完成**（cloudgpt_api.py 之前丢失，现已恢复）
- 需要：在服务器上跑 v7 精炼(~20min API)，然后传到 Windows 跑 ensemble 评估
- 代理图：217/220 已缓存
- CLIP features：363MB 已传到 Windows

### GeneCIS 4子集 Ensemble 验证
- v7 精炼缓存：4个子集都有
- 需要：在 Windows GPU 上跑 ensemble 评估（需要数据集图片做 per-query gallery）
- 代理图：change_obj 198, focus_obj 147, change_attr 200, focus_attr 200
- 阻塞：Windows 需要访问 COCO val2017(788MB) + Visual Genome(15GB) 的图片
- 方案：SSHFS 挂载已安装但在 SSH session 中不可用，需要在 Windows 桌面环境操作

---

## 三、方法概述

### 完整流程
```
第一轮: 参考图 + 修改文本 → MLLM(CoT) → 初始描述 D₁ → 文生图 → 代理图

第二轮: 参考图 + 代理图 + 修改文本 → MLLM(V7 Anti-Hallucination CoT) → 精炼描述 D₂

检索:   text_feat = normalize(0.7·CLIP(D₁) + 0.3·CLIP(D₂))    ← Description Ensemble
        score = 0.9·sim(text_feat, gallery) + 0.1·sim(CLIP(代理图), gallery)  ← 代理图后融合
```

### 三个创新点
1. **Visual Proxy（视觉代理）**: 文生图生成代理图，提供图像空间检索信号
2. **V7 Anti-Hallucination CoT**: 代理图仅做诊断工具，禁止引入AI幻觉细节
3. **Description Ensemble**: 融合原始+精炼描述的CLIP向量，保证不退化

### V7 Prompt 核心设计
- 代理图只用来CHECK是否正确，不能作为描述来源
- 禁止从代理图引入背景/材质/环境等细节
- 保留完整CoT（Thoughts → Reflections → Target Description）
- 强制短输出
- 实现: `src/refine_prompts.py` 中的 `V7_ANTI_HALLUCINATION`

### Ensemble 参数
- β=0.7: 原始描述权重70%，精炼描述30%
- α=0.9: 文本信号90%，代理图10%
- 数学解释见 METHOD_REPORT.md

---

## 四、Prompt 演进历史

| 版本 | 策略 | CIRR R@10 | shirt R@10 |
|:---:|------|:---:|:---:|
| original | 比较原图和代理图 | 64.5(-2.5) | 26.0(-4.0) |
| v5_cot_refine | CoT但未限制幻觉 | 65.5(-1.5) | — |
| v6_cot_context | 给MLLM第一轮输出 | 49.5(-17.5) | — |
| **v7** | 代理图只做诊断+短描述 | **73.5(+6.5)** | 26.0(-4.0) |
| v9 | 保留区分词 | 67.5(+0.5) | 29.5(-0.5) |
| v10 | v7+v9折中 | 66.0(-1.0) | 25.5(-4.5) |
| **v7+Ensemble** | v7+描述融合 | **69.0(+2.0)** | **30.5(+0.5)** |

---

## 五、缓存资产

### 代理图 (proxy_cache/)
| 数据集 | 数量 | 说明 |
|--------|:---:|------|
| fashioniq_dress | 1903 | 全量 |
| fashioniq_shirt | 200 | 随机200 |
| fashioniq_toptee | 200 | 随机200 |
| cirr | 195 | 随机200(5缺失) |
| circo | 217 | 全量 |
| genecis_change_object | 198 | 随机200 |
| genecis_focus_object | 147 | 随机200(53缺失) |
| genecis_change_attribute | 200 | 随机200 |
| genecis_focus_attribute | 200 | 随机200 |

### V7 精炼缓存 (outputs/prompt_ab_test/)
| 数据集 | 状态 |
|--------|:---:|
| fashioniq_dress | ✅ |
| fashioniq_shirt | ✅ |
| fashioniq_toptee | ✅ |
| cirr | ✅ |
| circo | ❌ 需要跑 |
| genecis_change_object | ✅ |
| genecis_focus_object | ✅ |
| genecis_change_attribute | ✅ |
| genecis_focus_attribute | ✅ |

### Windows D:\osrcir_remote\ 已有数据
- 评估脚本: win_ensemble_eval.py, win_eval_all.py
- Baseline 结果: dress, shirt, toptee, cirr, circo
- V7 精炼缓存: 全部9个数据集
- CLIP features: 全部5个(dress/shirt/toptee/cirr/circo)
- 代理图: dress(199), shirt(200), toptee(200), cirr(195), circo(217)

---

## 六、环境信息

### Linux 服务器
- IP: 1.15.92.20 (公网)
- 内存: 3.6GB RAM + 9GB swap
- Python 3.11, openai 2.29.0
- 数据集: FashionIQ, CIRR, CIRCO, GeneCIS 全部在 datasets/
- 代码: src/, 所有文件已恢复

### Windows 笔记本 (YangsY9000X)
- RTX 4060 Laptop GPU (8GB)
- 32GB RAM
- Python 3.11 (D:\env-py311), CLIP 已安装
- SSH 反向隧道: `ssh -R 2222:localhost:22 root@1.15.92.20`
- WinFsp + SSHFS-Win 已安装
- D:\osrcir_remote\ 有预传数据

### 关键文件恢复记录
- cloudgpt_api.py: 在本次session中丢失，已从 pyc 缓存恢复
- refine_prompts.py: 在本次session中多次丢失重建，当前只含 V7
- visual_proxy_combined.py, prompt_ab_test.py, eval_ensemble.py, genecis_combined.py:
  在本次session中创建但可能已丢失，需要重建

---

## 七、已花费

| 阶段 | 花费 |
|------|------|
| 论文复现 + dress/CIRCO 全量 | ~76元 |
| shirt/toptee/CIRR/GeneCIS 200样本 | ~49元 |
| Prompt A/B 测试 (v2/v5/v6/v7/v9/v10) | ~15元 |
| **累计** | **~140元** |

---

## 八、下一步

1. 跑 CIRCO v7 精炼（服务器，cloudgpt_api 已恢复）
2. 在 Windows GPU 上跑 CIRCO + GeneCIS ensemble 评估
3. 如果全部提升 → 选定 β=0.7 α=0.9 跑全量实验
4. 消融实验 + 论文撰写
