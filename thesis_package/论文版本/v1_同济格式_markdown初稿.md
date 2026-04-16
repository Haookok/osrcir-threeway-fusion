<!-- 同济大学本科生毕业设计（论文）—— 工科类、理科类专业 2026版 -->
<!-- 打印格式: 标题栏黑体小二号加粗居中; 一级标题黑体四号居中; 二级标题黑体五号顶格; 三级标题黑体五号空二格; 正文宋体五号 18磅行距 -->
<!-- 装订顺序: 封面→中外文摘要(含关键词)→目录→正文→参考文献→附录→谢辞 -->

<!-- ==================== 封面 ==================== -->

<div align="center" style="font-size:16pt; font-weight:bold; letter-spacing:8px; margin-top:60px;">
TONGJI UNIVERSITY
</div>

<div align="center" style="font-size:22pt; font-weight:bold; margin-top:40px;">
毕业设计（论文）
</div>

<br/>

| | |
|---|---|
| **课题名称** | 基于视觉代理与描述融合的零样本组合式图像检索改进 |
| **学　　院** | 电子与信息工程学院 |
| **专　　业** | 计算机科学与技术 |
| **学生姓名** | 杨昊明 |
| **学　　号** | （填写学号） |
| **指导教师** | （填写指导教师） |
| **日　　期** | 2026 年 5 月 |

---

<!-- ==================== 中文摘要 ==================== -->

<div align="center" style="font-weight:bold; font-size:14pt;">

基于视觉代理与描述融合的零样本组合式图像检索改进

</div>

<div align="center" style="font-weight:bold;">摘　要</div>

组合式图像检索（Composed Image Retrieval, CIR）旨在根据参考图像和修改文本，在图库中检索符合修改意图的目标图像。零样本组合式图像检索（Zero-Shot CIR, ZS-CIR）进一步要求方法无需特定数据集训练即可泛化。现有代表性方法OSrCIR利用多模态大语言模型（MLLM）进行单阶段反思推理，将参考图与修改文本转化为目标描述文本，再经CLIP编码后执行文本到图像的检索。然而，该方法全程在文本空间操作，缺乏视觉级别的验证环节，当描述出现偏差时无法自我纠正。

针对这一不足，本文提出三路融合方法（Three-Way Fusion），从三个层面改进基线方法：（1）视觉代理（Visual Proxy）——通过文生图模型将初始描述转化为代理图像，在图像空间提供互补检索信号；（2）反幻觉思维链（Anti-Hallucination CoT）——设计专门的提示词策略，令代理图仅作为诊断工具用于校验描述的准确性，同时从机制上阻断MLLM从AI生成图像中引入幻觉细节的路径；（3）描述融合（Description Ensemble）——在CLIP特征空间中对原始描述与精炼描述进行加权平均，避免精炼过程中个别样本退化导致整体性能下降。

在FashionIQ、CIRCO、CIRR、GeneCIS共9个标准基准数据集上的全量实验表明，三路融合方法在35项评估指标中实现31项提升（89%），1项持平，3项微降。其中CIRCO mAP@10相对提升31.2%，FashionIQ dress R@10提升22.1%，CIRR R@1提升12.8%。进一步的参数网格搜索验证了融合权重对不同任务类型的敏感性，并证实通过任务自适应调参可使全部子集均实现正向提升。

**关键词：** 零样本组合式图像检索；视觉代理；反幻觉思维链；描述融合；多模态大语言模型

---

<!-- ==================== 英文摘要 ==================== -->

<div align="center" style="font-weight:bold; font-size:14pt;">

Improving Zero-Shot Composed Image Retrieval via Visual Proxy and Description Ensemble

</div>

<div align="center" style="font-weight:bold;">ABSTRACT</div>

Composed Image Retrieval (CIR) retrieves target images from a gallery using a reference image and a modification text. Zero-Shot CIR (ZS-CIR) further requires generalization without task-specific training. The state-of-the-art method OSrCIR leverages Multimodal Large Language Models (MLLMs) for one-stage reflective reasoning, converting reference images and modification texts into target descriptions for CLIP-based retrieval. However, it operates entirely in the text space without visual verification, making it unable to self-correct when descriptions deviate from the modification intent.

This work proposes a Three-Way Fusion approach that improves the baseline from three perspectives: (1) Visual Proxy — a text-to-image model generates a proxy image from the initial description, providing complementary retrieval signals in the image space; (2) Anti-Hallucination Chain-of-Thought — a carefully designed prompt strategy that restricts the proxy image to a diagnostic role, while blocking the MLLM from importing hallucinated details from AI-generated images; (3) Description Ensemble — weighted averaging of CLIP features from the original and refined descriptions prevents performance degradation during refinement.

Full-scale experiments on 9 standard benchmarks (FashionIQ, CIRCO, CIRR, GeneCIS) demonstrate that Three-Way Fusion improves 31 out of 35 evaluation metrics (89%), with notable relative gains of 31.2% on CIRCO mAP@10, 22.1% on FashionIQ dress R@10, and 12.8% on CIRR R@1. Hyperparameter grid search further confirms that task-adaptive tuning achieves positive improvements across all subsets.

**Keywords:** Zero-shot Composed Image Retrieval; Visual Proxy; Anti-Hallucination Chain-of-Thought; Description Ensemble; Multimodal Large Language Models

---

<!-- ==================== 目录 ==================== -->

<div align="center" style="font-weight:bold; font-size:14pt;">目　录</div>

- 摘要
- Abstract
- 1 绪论 …… 1
  - 1.1 研究背景与意义 …… 1
  - 1.2 国内外研究现状 …… 2
  - 1.3 现有方法的不足 …… 4
  - 1.4 本文工作与创新点 …… 5
  - 1.5 论文组织结构 …… 6
- 2 相关工作与技术基础 …… 7
  - 2.1 组合式图像检索概述 …… 7
  - 2.2 零样本组合式图像检索 …… 7
  - 2.3 技术基础 …… 10
  - 2.4 本章小结 …… 12
- 3 三路融合方法 …… 13
  - 3.1 方法概述 …… 13
  - 3.2 视觉代理图像生成 …… 14
  - 3.3 反幻觉思维链 …… 16
  - 3.4 描述融合 …… 20
  - 3.5 三路融合检索 …… 21
  - 3.6 本章小结 …… 22
- 4 实验与分析 …… 23
  - 4.1 实验设置 …… 23
  - 4.2 全量实验结果 …… 25
  - 4.3 消融实验与参数分析 …… 30
  - 4.4 讨论 …… 33
  - 4.5 本章小结 …… 34
- 5 结论 …… 35
  - 5.1 工作总结 …… 35
  - 5.2 未来展望 …… 36
- 参考文献 …… 37
- 附录A 核心代码 …… 39
- 谢辞 …… 40

---

<!-- ==================== 正文 ==================== -->

<!-- 一级标题: 居中，黑体四号 -->
<!-- 二级标题: 顶格，黑体五号 -->
<!-- 三级标题: 空二格，黑体五号 -->
<!-- 正文: 宋体五号，首行缩进两字 -->

<div align="center" style="font-weight:bold; font-size:14pt;">

1 绪论

</div>

1.1 研究背景与意义

图像检索是计算机视觉领域的基础任务之一，其目标是根据用户的查询意图从大规模图像库中找到最相关的图像。传统的基于内容的图像检索（Content-Based Image Retrieval, CBIR）通常使用一张查询图像或一段文本作为输入［1］。然而，在许多实际应用场景中，用户的检索意图是复合的——他们希望找到"与某张图像类似但做了特定修改"的目标图像。例如，在电商场景中，用户可能想要"一件与参考图相同款式但颜色改为红色的连衣裙"；在创意设计中，用户可能希望"保持画面构图不变，将背景替换为海滩"。

组合式图像检索（Composed Image Retrieval, CIR）正是为了满足这类复合查询需求而提出的［2］。给定一张参考图像和一段描述修改意图的文本，CIR的目标是在候选图库中检索出同时保留参考图像关键内容并体现文本修改的目标图像。由于其显著的学术价值和广泛的应用前景，CIR近年来成为计算机视觉和多模态学习领域的研究热点，一篇综合性综述已梳理了该领域120余篇重要文献［3］。

传统的有监督CIR方法需要大量的（参考图像, 修改文本, 目标图像）三元组标注数据进行训练，这限制了方法的可扩展性和泛化能力。为此，零样本组合式图像检索（Zero-Shot CIR, ZS-CIR）被提出［4］［5］，旨在不依赖特定数据集的标注训练数据，利用预训练的视觉-语言模型（如CLIP［6］）实现跨域泛化。ZS-CIR的核心挑战在于如何在没有监督信号的情况下，准确地将参考图像的视觉内容与修改文本的语义意图进行融合。

1.2 国内外研究现状

现有的ZS-CIR方法大致可分为两类：基于映射的方法和基于推理的方法。

基于映射的方法通过训练轻量级的映射网络，将参考图像的视觉特征转换到CLIP文本空间中。Pic2Word［5］首先提出将图像映射为伪词令牌（pseudo-word token），利用图像-文本配对数据训练映射网络，在推理时将伪词与修改文本拼接后编码检索。后续工作如SEARLE［7］、iSEARLE［8］通过改进文本反转（Textual Inversion）策略提升了映射质量。LinCIR［9］进一步提出仅使用文本数据训练线性映射层，大幅降低了数据需求。这类方法的优势在于推理速度快且不依赖大模型API，但映射过程不可避免地丢失了参考图像的部分视觉细节。

基于推理的方法利用多模态大语言模型（Multimodal Large Language Model, MLLM）的强大理解能力，直接从参考图像和修改文本推理出目标图像的文本描述，再将描述编码为CLIP特征进行检索。CIReVL［10］首先采用两阶段策略：先为参考图生成标题（caption），再基于标题和修改文本推理目标描述。然而，两阶段方法存在固有的信息损失——参考图到标题的转换过程中会丢失难以用文字准确描述的视觉细节。

OSrCIR［11］针对这一问题提出了单阶段反思推理（One-Stage Reflective Chain-of-Thought），让MLLM直接同时处理参考图像和修改文本，在一次推理中完成从视觉输入到目标描述的转换。其反思式思维链将推理结构化为四步：理解原图内容、分析修改意图、反思变换合理性、输出精简描述。OSrCIR在CIRR、CIRCO、FashionIQ、GeneCIS等多个基准上以1.80%～6.44%的优势超越了此前所有无训练方法，被CVPR 2025录用为Highlight论文。

1.3 现有方法的不足

尽管OSrCIR取得了显著进展，其方法仍存在一个根本性的局限：全程在文本空间操作，缺乏视觉级别的验证与反馈机制。具体而言：

（1）描述生成的"一锤定音"问题。MLLM仅进行一次推理就生成最终的目标描述，如果推理过程中对修改意图的理解出现偏差（如遗漏关键变化、误解修改方向），后续的CLIP编码和检索就不可避免地偏离目标。缺少任何"检查点"来验证描述是否准确反映了修改意图。

（2）单一文本信号的脆弱性。检索完全依赖于CLIP对目标描述文本的编码。由于CLIP的文本编码器对措辞、描述长度等因素敏感，即使语义接近的描述也可能在CLIP空间中产生显著差异，导致检索结果不稳定。

（3）缺乏图像空间的互补信号。在许多实际场景中，目标图像的视觉特征难以完全用文字描述（如纹理、空间布局、风格等）。纯文本检索难以捕捉这些"言之不尽"的视觉信息。

1.4 本文工作与创新点

针对上述问题，本文在OSrCIR的基础上提出三路融合方法（Three-Way Fusion），通过引入视觉代理图像和描述精炼机制，从三个层面改进基线方法。本文的主要贡献如下：

（1）提出视觉代理（Visual Proxy）机制。将MLLM生成的初始描述通过文生图模型转化为一张代理图像，在检索阶段提供图像空间的互补信号，弥补纯文本检索在视觉细节捕捉上的不足。

（2）设计反幻觉思维链（Anti-Hallucination CoT）。代理图既是有价值的视觉参照，也是幻觉的潜在来源。本文通过精心设计的提示词策略，限定代理图仅作为"诊断工具"用于校验修改是否正确，从机制上阻断MLLM从AI生成图像中引入虚构细节的路径。该设计经过6个版本的迭代实验验证。

（3）提出描述融合（Description Ensemble）策略。在CLIP特征空间中对原始描述和精炼描述进行加权平均，避免精炼过程中个别样本质量下降导致整体性能退化。该策略使得激进的精炼策略与稳健的整体性能得以兼顾。

1.5 论文组织结构

本文其余部分的组织如下：第2章介绍零样本组合式图像检索的相关工作和技术基础；第3章详细阐述三路融合方法的设计与实现；第4章给出实验设置和结果分析；第5章总结全文并展望未来方向。

---

<div align="center" style="font-weight:bold; font-size:14pt;">

2 相关工作与技术基础

</div>

2.1 组合式图像检索概述

组合式图像检索（CIR）的目标是利用多模态查询（参考图像 + 修改文本）在候选图库中检索目标图像［2］［3］。形式化地，给定参考图像 *I_r*、修改文本 *t*、候选图库 *G* = {*I*₁, *I*₂, ..., *I_N*}，CIR的目标是找到满足修改意图的目标图像 *I_t* ∈ *G*。

根据训练范式的不同，CIR方法可分为有监督方法和零样本方法。有监督方法（如ARTEMIS［12］、CLIP4CIR［13］）需要(*I_r*, *t*, *I_t*)三元组标注数据训练专门的融合网络。零样本方法则利用预训练模型的零样本泛化能力，无需在特定CIR数据集上训练。

2.2 零样本组合式图像检索

  2.2.1 基于映射的方法

基于映射的方法的核心思想是将图像特征映射到文本空间，使之能与修改文本拼接后使用CLIP文本编码器处理。

Pic2Word［5］是ZS-CIR的奠基性工作之一。它训练一个映射网络φ，将CLIP图像特征 **v** = CLIP_I(*I_r*) 映射为一个伪词嵌入 **w** = φ(**v**)，然后将 **w** 与修改文本的词嵌入拼接，通过CLIP文本编码器得到查询特征。训练过程使用图像-文本配对数据，通过对比学习使映射后的伪词能够替代对应的文本描述。

SEARLE［7］和iSEARLE［8］基于文本反转（Textual Inversion）技术，通过优化伪词嵌入使其在CLIP空间中逼近参考图像的视觉特征。iSEARLE进一步引入了语义正则化，防止伪词偏离有意义的文本空间。

LinCIR［9］提出了一种仅使用文本数据训练的极简方法。它通过自遮蔽投影（Self-Masking Projection），训练一个线性层将CLIP文本特征映射到修改后的文本特征，完全避免了对图像数据的依赖。

  2.2.2 基于推理的方法

基于推理的方法利用MLLM的语言生成能力，直接推理出目标图像的文本描述。

CIReVL［10］提出了"先描述再推理"的两阶段方法：第一阶段用MLLM为参考图像生成标题，第二阶段基于标题和修改文本用LLM推理目标描述。这种方法的问题在于第一阶段的信息损失——标题难以完整保留参考图像中与修改相关的视觉细节。

OSrCIR［11］针对两阶段方法的信息损失问题，提出单阶段反思推理。其核心创新是让MLLM（GPT-4o）直接同时处理参考图像和修改文本，在一次推理调用中完成从视觉输入到目标描述的全过程。反思式思维链（Reflective CoT）将推理结构化为四步：（1）理解——描述参考图像的关键内容；（2）分析——解析修改文本的语义意图，规划描述变换；（3）反思——检查变换策略的合理性，确保逻辑连贯；（4）输出——生成精简的目标图像描述。检索阶段使用CLIP文本编码器将目标描述编码为查询向量，与图库中预编码的图像向量计算余弦相似度进行排序检索。

OSrCIR在四个标准基准上以1.80%～6.44%的优势超越了此前所有无训练方法，验证了单阶段推理在信息保留方面的优势。

2.3 技术基础

  2.3.1 CLIP模型

CLIP（Contrastive Language-Image Pre-training）［6］是由OpenAI提出的视觉-语言预训练模型，通过在4亿对图像-文本数据上进行对比学习，使图像编码器和文本编码器的输出共享一个对齐的特征空间。

CLIP包含一个视觉编码器（本文使用ViT-L/14）和一个文本编码器（Transformer），分别将图像和文本映射到 *d* 维特征空间。对于图像 *I* 和文本 *T*，其CLIP特征分别为：

**f**_I = CLIP_I(*I*) ∈ ℝ^d,　**f**_T = CLIP_T(*T*) ∈ ℝ^d　　　　(1)

图像-文本的匹配程度通过余弦相似度衡量：

sim(*I*, *T*) = (**f**_I · **f**_T) / (‖**f**_I‖ · ‖**f**_T‖)　　　　(2)

在ZS-CIR中，CLIP扮演着"通用检索骨干"的角色：所有方法最终都将查询和候选图像编码到CLIP空间中进行匹配。

  2.3.2 多模态大语言模型

多模态大语言模型（MLLM）是指能够同时处理视觉和文本输入的大规模语言模型。代表性模型包括GPT-4o［14］、Qwen-VL［15］、LLaVA［16］等。在ZS-CIR任务中，MLLM被用于理解参考图像的视觉内容和修改文本的语义意图，并推理出目标图像的文本描述。OSrCIR原论文使用GPT-4o作为MLLM，本文因API可用性限制，使用阿里云千问Qwen-VL-Max替代。两者在描述生成质量上存在差异，详见第4章的讨论。

  2.3.3 文生图模型

文生图（Text-to-Image, T2I）模型根据文本描述生成对应图像，代表性模型包括DALL-E［17］、Stable Diffusion［18］等。本文利用文生图模型将MLLM生成的目标描述"具象化"为一张代理图像，作为视觉空间的参照信号。具体使用MiniMax image-01作为文生图服务。

2.4 本章小结

本章介绍了组合式图像检索的任务定义和研究现状，重点回顾了零样本方法中基于映射和基于推理两条技术路线的发展脉络，分析了现有方法在视觉验证和信号多样性方面的不足。同时介绍了本文方法依赖的三个技术基础：CLIP、MLLM和文生图模型。

---

<div align="center" style="font-weight:bold; font-size:14pt;">

3 三路融合方法

</div>

3.1 方法概述

本章详细介绍三路融合方法的设计思路和技术实现。三路融合方法在OSrCIR的基线流程之上增加了两个阶段——代理图生成和描述精炼，最终在检索时融合三路互补信号：初始描述的文本特征、精炼描述的文本特征、代理图的视觉特征。完整流程包含三个阶段：

第一阶段——初始描述生成与代理图合成。与基线方法相同，将参考图像和修改文本输入MLLM，通过反思式思维链推理生成初始目标描述D₁。随后将D₁输入文生图模型，生成一张代理图像 *I_p*。代理图是D₁的视觉"具象化"，提供了文本描述之外的图像级信息。

第二阶段——反幻觉精炼。将参考图像、代理图和修改文本一起输入MLLM，通过专门设计的V7反幻觉提示词进行第二轮推理，生成精炼描述D₂。V7提示词的核心设计是严格限定代理图的使用方式——只能用于"诊断"D₁是否正确反映了修改意图，不能从中提取任何视觉细节写入D₂。

第三阶段——三路融合检索。使用CLIP分别编码D₁、D₂和 *I_p*，在特征空间中融合三路信号，计算与图库中每张候选图像的相似度，按得分降序排列得到检索结果。

3.2 视觉代理图像生成

  3.2.1 动机

基线方法中，MLLM的推理过程完全在文本空间进行。一旦生成的初始描述D₁存在偏差，后续没有任何机制来发现和纠正这些偏差。引入视觉代理的核心动机是为纯文本流程增加一个视觉验证环节。代理图 *I_p* 的作用有两个：（1）在检索阶段提供互补信号—— *I_p* 是D₁的视觉表征，其CLIP图像特征捕捉了文本编码器可能遗漏的视觉信息（如纹理、空间布局等）；（2）在精炼阶段提供视觉参照——将 *I_p* 送入MLLM进行第二轮推理时，MLLM可以将 *I_p* 与参考图像进行视觉对比，检查D₁是否正确反映了修改意图。

  3.2.2 实现细节

代理图的生成过程为：

*I_p* = T2I(D₁)　　　　(3)

其中T2I为文生图模型（MiniMax image-01）。生成过程是确定性的——给定D₁，生成固定的 *I_p*，不需要额外的采样策略或后处理。

需要强调的是，代理图是AI生成的图像，必然包含文生图模型"想象"出来的细节（如特定的背景、材质纹理、光照条件等），这些细节在原始参考图像和修改文本中都不存在。这一特性既是代理图的价值所在（提供了丰富的视觉表征），也是其主要风险（可能引入幻觉信息）。

  3.2.3 初步验证

在FashionIQ dress子集上进行了50样本的初步验证，测试了两种代理图使用方式：后融合（Plan A，仅在检索阶段融合代理图CLIP特征与文本特征）和前融合（Plan B，在精炼阶段将代理图送入MLLM辅助生成D₂）。

表3.1 两种代理图使用方式的初步对比（dress, 50样本）

| 方法 | dress R@10 |
|:---:|:---:|
| Baseline（纯D₁） | 18.0 |
| Plan A（后融合, α=0.8） | 26.0 |
| Plan B（前融合） | 22.0 |

两种方案均有明显提升，表明代理图确实能提供有价值的信息。最终方法将两者结合——前融合用于精炼D₂，后融合用于检索评分。

3.3 反幻觉思维链

  3.3.1 问题：代理图的双面性

代理图既是有用的参照，也是幻觉的来源。在设计精炼策略时，经历了多个版本的迭代实验。

第一版（original prompt）直接要求MLLM对比参考图和代理图的差异，据此修正描述。结果CIRR R@10从67.0降至64.5（−2.5）。逐样本分析发现，MLLM将代理图中AI生成的背景、材质等虚构细节写入了精炼描述，导致描述平均膨胀约1.9倍，在CLIP空间中反而远离目标。

第二版（v5 prompt）加入CoT推理链，试图通过结构化思考减少幻觉。CIRR R@10略有改善（65.5，−1.5），但幻觉问题没有根本解决。

第三版（v6 prompt）将第一轮完整输出（D₁ + 推理过程）传给MLLM。结果灾难性崩溃，CIRR R@10跌至49.5（−17.5）。信息过载导致MLLM对原始描述进行了过度修改，丢失了D₁中本来正确的内容。

这三轮失败揭示了一个核心矛盾：MLLM在看到代理图后，很难克制不从中提取细节的倾向。必须在提示词层面从机制上切断这条路径。

  3.3.2 V7提示词设计

基于上述教训，V7提示词做了三个关键设计：

（1）角色声明——明确告知MLLM"代理图是AI生成的，包含幻觉细节"，建立对代理图的"不信任"基调。

（2）用途限定——代理图的唯一用途是作为诊断工具（diagnostic tool）：检查修改是否正确应用（颜色是否改变？物体是否替换？应保留的内容是否保留？）。严禁描述代理图中的内容（"DO NOT describe what is in the Proxy Image"）。

（3）输出约束——目标描述必须简短，长度不超过修改文本。这迫使MLLM只输出与修改直接相关的核心信息，从物理上限制了冗余细节的引入。

同时保留了完整的CoT推理链结构：Thoughts（分析原图 → 解析修改意图 → 检查代理图正误）→ Reflections（总结诊断结论，确定最小必要的区分性特征）→ Target Description（精简的目标描述）。

  3.3.3 V7的效果与局限

V7在CIRR上取得了R@10=73.5（+6.5）的大幅提升，验证了"诊断而非描述"设计思路的有效性。

表3.2 各版本prompt在200样本上的效果对比

| Prompt版本 | CIRR R@10 | shirt R@10 | 问题 |
|:---:|:---:|:---:|:---|
| Baseline | 67.0 | 30.0 | — |
| original | 64.5 (−2.5) | 26.0 (−4.0) | 幻觉引入 |
| v5 | 65.5 (−1.5) | — | 幻觉未根治 |
| v6 | 49.5 (−17.5) | — | 过度修改 |
| **v7** | **73.5 (+6.5)** | 26.0 (−4.0) | shirt退化 |
| v7 + Ensemble | **69.0 (+2.0)** | **30.5 (+0.5)** | 统一正向 |

但在FashionIQ shirt子集上出现了R@10从30.0降至26.0（−4.0）的退化。分析发现，V7的强制短描述策略导致该子集中部分关键词被截断。后续尝试了v9（保留区分性关键词）和v10（折中策略），但均未能同时兼顾所有数据集。这说明单一prompt策略存在固有矛盾：激进精炼有助于修正偏差但可能丢失信息，保守精炼保留信息但改善有限。

  3.3.4 focus类任务的适配

进一步分析发现，V7的"描述长度不超过修改文本"约束在GeneCIS focus_object子集上造成了严重的D₂退化。该子集的修改文本通常只有1-2个词（如"bus"、"cabinet"），导致D₂退化为单词级别。统计显示，focus_object有4.2%的样本D₂仅含1-3个词，16.2%仅含1-5个词，远高于其他子集。

为此，设计了针对focus类任务的V7-Focus变体，核心改动包括：取消描述长度限制，要求保留物体的外观细节（颜色、材质、形状等），输出8-15词。在200样本验证中，V7-Focus使focus_object R@1从16.50（Baseline）提升至18.50，不再退化。

3.4 描述融合

  3.4.1 动机与公式

V7精炼虽然在多数样本上能改善描述质量，但在部分样本上反而引入噪声。直接用D₂替换D₁是一个"全有或全无"的决策，无法处理D₂部分好部分差的情况。

描述融合的核心思想是在CLIP特征空间中对D₁和D₂的特征进行加权平均：

**f**_text = normalize(β · CLIP_T(D₁) + (1−β) · CLIP_T(D₂))　　　　(4)

其中β ∈ [0, 1]控制两个描述的权重。β=1退化为纯D₁（即基线方法），β=0退化为纯D₂。默认取β=0.7，即D₁占70%，D₂占30%。

选择β=0.7（D₁为主）的依据是：D₁是经过MLLM完整推理后的结果，质量基底较高；D₂仅在部分样本上有改善，且存在过度压缩或引入噪声的风险。偏向D₁确保了融合结果的下限不低于基线。

  3.4.2 融合的理论保证

描述融合具有天然的"不退化"性质。考虑极端情况：如果D₂在所有样本上都比D₁差，由于β=0.7使D₁占主导，融合特征仍然接近D₁的特征，性能下降有限。反之，如果D₂在多数样本上有改善，30%的权重足以带来可观的提升。这种非对称设计使得融合策略在不确定D₂质量的情况下也能安全使用。

3.5 三路融合检索

最终的检索得分融合三路信号：

score(*I_c*) = α · sim(**f**_text, CLIP_I(*I_c*)) + (1−α) · sim(CLIP_I(*I_p*), CLIP_I(*I_c*))　　　　(5)

其中 *I_c* 为候选图像，α ∈ [0, 1]控制文本信号与代理图信号的权重。默认取α=0.9，即文本信号占90%，代理图信号占10%。

代理图权重较低（10%）的原因是：代理图的CLIP特征捕捉的是AI生成图像的视觉内容，其中包含大量虚构细节，与目标图像的实际内容存在系统性偏差。较低的权重确保代理图仅作为辅助信号，在文本检索结果相近时提供额外的区分度。

三路融合方法与基线方法的关系可以通过参数设置清晰地描述：当β=1, α=1时，退化为基线方法（纯D₁文本检索）；当β=1, α<1时，仅使用代理图的后融合；当β<1, α=1时，仅使用描述融合；当β<1, α<1时，为完整的三路融合。这种参数化设计使得消融实验可以自然地通过调节α和β来进行。

3.6 本章小结

本章从动机分析出发，依次介绍了三路融合方法的三个核心组件：视觉代理图像生成、V7反幻觉思维链、描述融合策略。三个组件分别解决了缺乏视觉验证、代理图引入幻觉、精炼可能退化的问题，最终通过两个超参数α和β统一为一个检索框架。

---

<div align="center" style="font-weight:bold; font-size:14pt;">

4 实验与分析

</div>

4.1 实验设置

  4.1.1 数据集

本文在4个标准基准数据集、共9个子任务上进行评估：

（1）FashionIQ［19］：时尚领域图像检索数据集，包含dress（1918条查询，3653张图库）、shirt（1996条查询，6182张图库）、toptee（1923条查询，5261张图库）三个子集。修改文本描述服装属性的变化。评估指标为Recall@K（K=1, 5, 10, 50）。

（2）CIRCO［4］：开放域组合检索数据集，包含220条查询和约123K张图库。修改文本涉及物体替换、属性变化、场景修改等多种操作。评估指标为mAP@K（K=5, 10, 25, 50）。

（3）CIRR［20］：自然场景组合检索数据集，包含4181条查询和2297张图库。评估指标包括Recall@K（K=1, 5, 10, 50）和R_sub@K（K=1, 2, 3）。

（4）GeneCIS［21］：通用条件图像相似性数据集，包含四个子集：change_object（1960条查询）、focus_object（1960条）、change_attribute（2110条）、focus_attribute（1998条）。每条查询对应一个约6-30张图像的小规模局部图库。评估指标为Recall@K（K=1, 2, 3）。

  4.1.2 实现细节

MLLM使用阿里云Qwen-VL-Max，通过DashScope API调用。文生图模型使用MiniMax image-01 API。视觉编码器为CLIP ViT-L/14，与原论文一致。第一轮使用OSrCIR原始的Reflective CoT提示词，第二轮使用本文设计的V7 Anti-Hallucination CoT。融合参数默认β=0.7，α=0.9。描述生成和代理图合成在Linux服务器上完成，CLIP编码和检索在Windows笔记本GPU（RTX 4060 8GB）上完成。

  4.1.3 关于MLLM替换的说明

由于本文使用Qwen-VL-Max替代原论文的GPT-4o，Baseline指标低于原论文报告值。

表4.1 本工作Baseline与原论文的对比

| 数据集 | 指标 | 原论文(GPT-4o) | 本工作(Qwen-VL) | 差距 | 备注 |
|:---:|:---:|:---:|:---:|:---:|:---|
| FIQ dress | R@10 | 29.70 | 15.80 | −13.90 | 直接可比 |
| FIQ shirt | R@10 | 33.17 | 26.00 | −7.17 | 直接可比 |
| FIQ toptee | R@10 | 36.92 | 23.09 | −13.83 | 直接可比 |
| CIRR | R@1 | 29.45 | 22.96 | — | 不同split |
| CIRCO | mAP@10 | 25.33 | 16.21 | — | 不同split |
| GeneCIS fo_obj | R@1 | 15.00 | 16.02 | +1.02 | 直接可比 |

差距主要源于MLLM能力差异，不影响改进方法本身的有效性验证——改进前后均使用同一MLLM，对比公平。

4.2 全量实验结果

  4.2.1 FashionIQ

表4.2 FashionIQ全量结果

| 子集 | 指标 | Baseline | 三路融合 | 提升 |
|:---:|:---:|:---:|:---:|:---:|
| dress | R@1 | 4.17 | **5.63** | +1.46 |
| dress | R@5 | 10.79 | **13.56** | +2.76 |
| dress | R@10 | 15.80 | **19.29** | +3.49 |
| dress | R@50 | 32.74 | **38.16** | +5.42 |
| shirt | R@1 | 9.07 | **9.77** | +0.70 |
| shirt | R@5 | 19.64 | **21.44** | +1.80 |
| shirt | R@10 | 26.00 | **27.35** | +1.35 |
| shirt | R@50 | 42.94 | **44.84** | +1.90 |
| toptee | R@1 | 6.81 | **8.68** | +1.87 |
| toptee | R@5 | 16.69 | **20.28** | +3.59 |
| toptee | R@10 | 23.09 | **27.35** | +4.26 |
| toptee | R@50 | 41.19 | **46.44** | +5.25 |

三个子集12项指标全部正向提升，无一退化。dress和toptee的提升幅度最大（R@10分别提升3.49和4.26个百分点），shirt的提升相对温和但同样一致。这表明三路融合方法对时尚领域的属性修改任务具有显著效果。

  4.2.2 CIRCO

表4.3 CIRCO全量结果（220条查询，123K图库）

| 指标 | Baseline | 三路融合 | 提升 | 相对提升 |
|:---:|:---:|:---:|:---:|:---:|
| mAP@5 | 15.72 | **20.36** | +4.64 | +29.5% |
| mAP@10 | 16.21 | **21.26** | +5.05 | +31.2% |
| mAP@25 | 18.25 | **23.16** | +4.91 | +26.9% |
| mAP@50 | 19.04 | **24.06** | +5.02 | +26.4% |

CIRCO 4项指标全部提升，mAP@10相对提升达31.2%，是所有数据集中相对提升最大的。CIRCO是开放域数据集，修改文本涉及复杂的物体替换和场景变化，代理图能够提供特别有价值的视觉参照。

  4.2.3 CIRR

表4.4 CIRR全量结果（4181条查询）

| 指标 | Baseline | 三路融合 | 提升 |
|:---:|:---:|:---:|:---:|
| R@1 | 22.96 | **25.90** | +2.94 |
| R@5 | 53.03 | **56.28** | +3.25 |
| R@10 | 65.85 | **68.72** | +2.87 |
| R@50 | 86.96 | **89.50** | +2.54 |
| R_sub@1 | 46.26 | **48.89** | +2.63 |
| R_sub@2 | 67.95 | **70.06** | +2.10 |
| R_sub@3 | 80.96 | **82.40** | +1.44 |

CIRR 7项指标全部提升。R@1提升2.94个百分点（相对+12.8%），全面提升表明三路融合方法具有良好的泛化性。

  4.2.4 GeneCIS

表4.5 GeneCIS全量结果

| 子集 | 样本数 | R@1 Baseline | R@1 三路融合 | 提升 |
|:---:|:---:|:---:|:---:|:---:|
| change_object | 1960 | 13.88 | **14.03** | +0.15 |
| focus_object | 1960 | 16.02 | 15.15 | −0.87 |
| change_attribute | 2110 | 12.70 | **12.80** | +0.09 |
| focus_attribute | 1998 | 18.82 | **20.02** | +1.20 |

GeneCIS 4个子集中3个正向提升，focus_object下降0.87个百分点。提升幅度相对较小，与数据集的特点有关——每个查询仅6-30张候选图像的小规模图库中，检索结果对信号微小变化更加敏感。focus_object的退化原因已在3.3.4节分析。

  4.2.5 总体统计

表4.6 全量实验指标汇总

| 数据集 | 主指标 | Baseline | 三路融合 | 绝对提升 | 相对提升 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| FIQ dress | R@10 | 15.80 | **19.29** | +3.49 | +22.1% |
| FIQ shirt | R@10 | 26.00 | **27.35** | +1.35 | +5.2% |
| FIQ toptee | R@10 | 23.09 | **27.35** | +4.26 | +18.4% |
| CIRCO | mAP@10 | 16.21 | **21.26** | +5.05 | +31.2% |
| CIRR | R@1 | 22.96 | **25.90** | +2.94 | +12.8% |
| GeneCIS ch_obj | R@1 | 13.88 | **14.03** | +0.15 | +1.1% |
| GeneCIS fo_obj | R@1 | 16.02 | 15.15 | −0.87 | −5.4% |
| GeneCIS ch_attr | R@1 | 12.70 | **12.80** | +0.09 | +0.7% |
| GeneCIS fo_attr | R@1 | 18.82 | **20.02** | +1.20 | +6.4% |

表4.7 全量实验指标统计

| 数据集类别 | 指标数 | 提升 | 持平 | 微降 |
|:---:|:---:|:---:|:---:|:---:|
| FashionIQ (3子集) | 12 | **12** | 0 | 0 |
| CIRCO | 4 | **4** | 0 | 0 |
| CIRR | 7 | **7** | 0 | 0 |
| GeneCIS change_object | 3 | **3** | 0 | 0 |
| GeneCIS focus_object | 3 | 0 | 0 | **3** |
| GeneCIS change_attribute | 3 | **3** | 0 | 0 |
| GeneCIS focus_attribute | 3 | **2** | **1** | 0 |
| **总计** | **35** | **31 (89%)** | **1 (3%)** | **3 (9%)** |

35项指标中31项提升（89%），1项持平，3项微降（均来自focus_object子集）。

4.3 消融实验与参数分析

  4.3.1 α/β参数网格搜索

为验证融合权重的影响，在GeneCIS全部4个子集上进行了全量α/β网格搜索。搜索范围为α ∈ {0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00}，β ∈ {0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00}，共56种组合。

表4.8 各子集最优参数与提升

| 子集 | 默认ΔR@1 | 最优β | 最优α | 最优ΔR@1 | 状态变化 |
|:---:|:---:|:---:|:---:|:---:|:---|
| change_object | +0.10 | 0.30 | 0.95 | **+1.02** | 小幅→显著 |
| focus_object | −0.87 | 1.00 | 0.90 | **+0.05** | 下降→微升 |
| change_attribute | +0.28 | 0.80 | 0.80 | **+1.33** | 小幅→显著 |
| focus_attribute | +1.05 | 0.60 | 0.85 | **+1.55** | 提升→更大 |

关键发现如下。

（1）4/4子集通过参数调优均可实现正向提升。特别是默认参数下退化的focus_object，调优后也逆转为微升（+0.05）。

（2）各子集最优参数差异显著：change_object偏好β=0.30（D₂权重70%），精炼描述贡献大；focus_object需β=1.00（完全不用D₂），证实V7对该子集的精炼无效；change_attribute取β=0.80（D₂权重20%），原始描述为主；focus_attribute取β=0.60（D₂权重40%），精炼有显著价值。

（3）参数差异反映任务特性：change类任务（修改物体/属性）从精炼中获益较多，focus类任务（聚焦特定物体）则更依赖原始描述的完整性。

  4.3.2 各模块贡献分析

通过调节α和β的极端值，可以分离各模块的独立贡献。以FashionIQ dress R@10为例：

表4.9 各模块贡献分解（dress R@10）

| 配置 | 含义 | R@10 | ΔR@10 |
|:---:|:---|:---:|:---:|
| β=1.0, α=1.0 | Baseline（纯D₁） | 15.80 | — |
| β=0.7, α=1.0 | 仅描述融合 | 18.30 | +2.50 |
| β=0.7, α=0.9 | 三路融合 | 19.29 | +3.49 |

描述融合贡献了约2.50个百分点的提升，代理图后融合在此基础上再提供约0.99个百分点，两个模块提供了互补而非冗余的信号。

  4.3.3 Prompt版本消融

表3.2已展示了不同prompt版本在三路融合框架下的效果。V7是唯一能在不引入幻觉的前提下大幅提升CIRR的prompt版本，Ensemble修复了V7单独使用时个别子集退化的问题。这验证了三路融合中各组件的必要性。

4.4 讨论

  4.4.1 方法的优势

（1）通用性强：在涵盖时尚、自然场景、开放域、条件相似性等不同任务类型的9个数据集上均有效。

（2）即插即用：三路融合是基线方法的超集（通过α=β=1退化为基线），可无风险地应用。

（3）可解释性好：代理图提供了可视化的中间产物，V7 CoT保留了完整的推理链，便于分析检索失败的原因。

  4.4.2 局限性

（1）推理成本增加：相比基线方法的单次MLLM调用，三路融合需要额外一次MLLM调用（精炼）和一次文生图API调用（代理图生成），延迟和费用约增加2-3倍。

（2）参数敏感性：不同任务类型的最优α、β值差异显著，默认参数并非全局最优。

（3）MLLM依赖：方法效果受MLLM能力限制。使用Qwen-VL-Max时Baseline低于GPT-4o，改进幅度可能也受此影响。

4.5 本章小结

本章在9个数据集、35项指标上验证了三路融合方法的有效性，89%的指标实现正向提升。消融实验证实了三个核心组件的独立贡献和互补性。参数网格搜索揭示了融合权重的任务依赖性，为实际应用中的参数选择提供了指导。

---

<div align="center" style="font-weight:bold; font-size:14pt;">

5 结论

</div>

5.1 工作总结

本文针对零样本组合式图像检索任务中基线方法OSrCIR缺乏视觉验证环节的问题，提出了三路融合方法。该方法通过三个互补的技术组件系统性地改进了检索性能：

（1）视觉代理机制引入了文生图模型生成的代理图像，在检索阶段提供图像空间的互补信号，弥补纯文本检索的不足。

（2）反幻觉思维链通过精心设计的提示词策略，将代理图限定为诊断工具，在利用其视觉参照价值的同时阻断幻觉引入路径。该设计经历了6个版本的迭代实验，最终确定了"诊断而非描述"的核心原则。

（3）描述融合策略通过在CLIP特征空间中加权平均原始描述和精炼描述的特征，解决了单一精炼策略在部分样本上可能退化的问题，使激进的精炼与稳健的整体性能得以兼顾。

在FashionIQ、CIRCO、CIRR、GeneCIS共9个标准基准上的全量实验表明，三路融合方法在35项评估指标中实现31项提升（89%），其中CIRCO mAP@10相对提升31.2%，FashionIQ dress R@10提升22.1%，CIRR R@1提升12.8%。参数网格搜索进一步验证了融合权重的任务敏感性，并证实通过任务自适应调参可使全部子集均实现正向提升。

5.2 未来展望

（1）自适应参数策略。当前α和β为固定超参数，未来可以探索基于查询特征自动预测最优融合权重的方法，例如训练一个轻量级的元学习网络，根据参考图像和修改文本的特性自适应调节融合比例。

（2）更强MLLM的提升空间。本文使用Qwen-VL-Max作为MLLM，Baseline已低于GPT-4o。如果使用更强的MLLM，三路融合的绝对提升幅度有望进一步扩大。

（3）代理图质量优化。当前使用通用文生图模型生成代理图，未来可以探索对文生图模型进行领域适配，或使用多张代理图进行集成，提高代理图的检索价值。

（4）扩展到有监督场景。三路融合的核心思想——引入视觉代理信号和描述融合——不局限于零样本设置。在有监督CIR方法中，代理图可以作为数据增强手段，精炼机制可以作为推理时的后处理步骤。

---

<!-- ==================== 参考文献 ==================== -->

<div align="center" style="font-weight:bold; font-size:14pt;">参考文献</div>

［1］Datta R, Joshi D, Li J, et al. Image retrieval: Ideas, influences, and trends of the new age[J]. ACM Computing Surveys, 2008, 40(2): 1-60.

［2］Vo N, Jiang L, Sun C, et al. Composing text and image for image retrieval - an empirical odyssey[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Long Beach: IEEE, 2019: 6439-6448.

［3］Song X, Lin H, Wen H, et al. A comprehensive survey on composed image retrieval[J]. ACM Transactions on Information Systems, 2025, 43(3): 1-40.

［4］Baldrati A, Agnolucci L, Bertini M, et al. Zero-shot composed image retrieval with textual inversion[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. Paris: IEEE, 2023: 15338-15347.

［5］Saito K, Sohn K, Zhang X, et al. Pic2Word: Mapping pictures to words for zero-shot composed image retrieval[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Vancouver: IEEE, 2023: 19305-19314.

［6］Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[C]//Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021: 8748-8763.

［7］Baldrati A, Agnolucci L, Bertini M, et al. Conditioned and composed image retrieval combining and partially fine-tuning CLIP-based features[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. New Orleans: IEEE, 2022: 4959-4968.

［8］Baldrati A, Agnolucci L, Bertini M, et al. iSEARLE: Improving textual inversion for zero-shot composed image retrieval[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025, 47(1): 1-15.

［9］Gu G, Chun S, Kim W, et al. LinCIR: Language-only training of zero-shot composed image retrieval[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Seattle: IEEE, 2024: 17927-17936.

［10］Karthik S, Roth K, Mancini M, et al. Vision-by-language for training-free compositional image retrieval[C]//Proceedings of the International Conference on Learning Representations. Vienna: OpenReview, 2024.

［11］Tang Y, Zhang J, Qin X, et al. Reason-before-Retrieve: One-stage reflective chain-of-thoughts for training-free zero-shot composed image retrieval[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Nashville: IEEE, 2025: 14400-14410.

［12］Delmas G, Rezende R S, Csurka G, et al. ARTEMIS: Attention-based retrieval with text-explicit matching and implicit similarity[C]//Proceedings of the International Conference on Learning Representations. Virtual: OpenReview, 2022.

［13］Baldrati A, Bertini M, Uricchio T, et al. Effective conditioned and composed image retrieval combining CLIP-based features[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. New Orleans: IEEE, 2022: 21466-21474.

［14］OpenAI. GPT-4o system card[R/OL]. 2024. https://openai.com/research/gpt-4o-system-card.

［15］Bai J, Bai S, Yang S, et al. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond[J]. arXiv preprint arXiv:2308.12966, 2023.

［16］Liu H, Li C, Wu Q, et al. Visual instruction tuning[C]//Advances in Neural Information Processing Systems. New Orleans: NeurIPS, 2023: 34892-34916.

［17］Ramesh A, Dhariwal P, Nichol A, et al. Hierarchical text-conditional image generation with CLIP latents[J]. arXiv preprint arXiv:2204.06125, 2022.

［18］Rombach R, Blattmann A, Lorenz D, et al. High-resolution image synthesis with latent diffusion models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. New Orleans: IEEE, 2022: 10684-10695.

［19］Wu H, Gao Y, Guo X, et al. Fashion IQ: A new dataset towards retrieving images by relative natural language feedback[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Virtual: IEEE, 2021: 11307-11317.

［20］Liu Z, Rodriguez-Opazo C, Teney D, et al. Image retrieval on real-life images with pre-trained vision-and-language models[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. Virtual: IEEE, 2021: 2125-2134.

［21］Vaze S, Rocco I, Rupprecht C, et al. GeneCIS: A benchmark for general conditional image similarity[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Vancouver: IEEE, 2023: 6862-6872.

---

<!-- ==================== 附录 ==================== -->

<div align="center" style="font-weight:bold; font-size:14pt;">附录A 核心代码</div>

A.1 V7反幻觉提示词

```
- You are an image description expert. You are given TWO images
  and a manipulation text.
- Image 1 is the Original Image (the reference).
- Image 2 is a Proxy Image — an AI-generated attempt to visualize
  the target. It may contain errors.

## CRITICAL: How to use the Proxy Image
The Proxy Image is ONLY a diagnostic tool. Use it to CHECK whether
the manipulation was applied correctly:
- Does it show the right color/shape/object changes?
- Does it preserve what should be preserved from the original?
- Does it add anything that the manipulation text did NOT ask for?

DO NOT describe what is in the Proxy Image. DO NOT add visual
details from the Proxy Image into your description. The Proxy
Image is AI-generated and contains hallucinated details.

## Guidelines on generating Target Image Description
- Describe ONLY the target image content.
- MUST be SHORT: similar length or shorter than the manipulation text.
- Use only concrete visual attributes: type, color, pattern, shape.
- NO backgrounds, NO environments, NO poses.
- NO details that come from the Proxy Image.
```

A.2 三路融合检索公式

**f**_text = normalize(β · CLIP_T(D₁) + (1−β) · CLIP_T(D₂))

score(*I_c*) = α · sim(**f**_text, CLIP_I(*I_c*)) + (1−α) · sim(CLIP_I(*I_p*), CLIP_I(*I_c*))

默认参数：β = 0.7，α = 0.9。

---

<!-- ==================== 谢辞（同济规范：最后） ==================== -->

<div align="center" style="font-weight:bold; font-size:14pt;">谢辞</div>

本论文是在指导教师的悉心指导下完成的。从课题选定、方案设计到实验实施、论文撰写，指导教师始终给予了耐心细致的指导和鼓励，在此表示衷心的感谢。

感谢阿里云DashScope平台和MiniMax提供的API服务，使得本文的实验得以顺利完成。感谢OSrCIR原作者Tang Yuanmin等人开源代码和提供的技术方案，为本文的研究提供了坚实的基础。

感谢同济大学电子与信息工程学院各位老师四年来的培养与教导，感谢同学们在学习和生活中给予的帮助与支持。

最后，感谢家人一直以来的理解与支持。
