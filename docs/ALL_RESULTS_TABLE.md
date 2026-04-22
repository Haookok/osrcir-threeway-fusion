# 三路融合全量实验结果汇总

> 方法: Qwen-VL-Max + V7 Anti-Hallucination CoT + Description Ensemble + Visual Proxy, CLIP ViT-L/14
> GeneCIS 使用专用 prompt + best α/β；其余使用 β=0.7, α=0.9


| 数据集       | 子集               | 指标      | Baseline (D₁) | 三路融合      | 绝对提升   | 相对提升   |
| --------- | ---------------- | ------- | ------------- | --------- | ------ | ------ |
| FashionIQ | dress            | R@1     | 4.17          | **5.63**  | +1.46  | +35.0% |
| FashionIQ | dress            | R@5     | 10.79         | **13.56** | +2.76  | +25.6% |
| FashionIQ | dress            | R@10    | 15.80         | **19.29** | +3.49  | +22.1% |
| FashionIQ | dress            | R@50    | 32.74         | **38.16** | +5.42  | +16.6% |
| FashionIQ | shirt            | R@1     | 9.07          | **9.77**  | +0.70  | +7.7%  |
| FashionIQ | shirt            | R@5     | 19.64         | **21.44** | +1.80  | +9.2%  |
| FashionIQ | shirt            | R@10    | 26.00         | **27.35** | +1.35  | +5.2%  |
| FashionIQ | shirt            | R@50    | 42.94         | **44.84** | +1.90  | +4.4%  |
| FashionIQ | toptee           | R@1     | 6.81          | **8.68**  | +1.87  | +27.5% |
| FashionIQ | toptee           | R@5     | 16.69         | **20.28** | +3.59  | +21.5% |
| FashionIQ | toptee           | R@10    | 23.09         | **27.35** | +4.26  | +18.4% |
| FashionIQ | toptee           | R@50    | 41.19         | **46.44** | +5.25  | +12.7% |
| CIRCO     | —                | mAP@5   | 15.72         | **20.36** | +4.64  | +29.5% |
| CIRCO     | —                | mAP@10  | 16.21         | **21.26** | +5.05  | +31.2% |
| CIRCO     | —                | mAP@25  | 18.25         | **23.16** | +4.91  | +26.9% |
| CIRCO     | —                | mAP@50  | 19.04         | **24.06** | +5.02  | +26.4% |
| CIRR      | —                | R@1     | 22.96         | **25.90** | +2.94  | +12.8% |
| CIRR      | —                | R@5     | 53.03         | **56.28** | +3.25  | +6.1%  |
| CIRR      | —                | R@10    | 65.85         | **68.72** | +2.87  | +4.4%  |
| CIRR      | —                | R@50    | 86.96         | **89.50** | +2.54  | +2.9%  |
| CIRR      | —                | R_sub@1 | 46.26         | **48.89** | +2.63  | +5.7%  |
| CIRR      | —                | R_sub@2 | 67.95         | **70.06** | +2.10  | +3.1%  |
| CIRR      | —                | R_sub@3 | 80.96         | **82.40** | +1.44  | +1.8%  |
| GeneCIS   | change_object    | R@1     | 13.83         | **25.51** | +11.68 | +84.5% |
| GeneCIS   | change_object    | R@2     | 25.20         | **40.26** | +15.06 | +59.8% |
| GeneCIS   | change_object    | R@3     | 36.22         | **50.77** | +14.55 | +40.2% |
| GeneCIS   | focus_object     | R@1     | 16.02         | **23.62** | +7.60  | +47.4% |
| GeneCIS   | focus_object     | R@2     | 26.58         | **37.04** | +10.46 | +39.4% |
| GeneCIS   | focus_object     | R@3     | 35.61         | **47.45** | +11.84 | +33.2% |
| GeneCIS   | change_attribute | R@1     | 12.65         | **21.79** | +9.14  | +72.3% |
| GeneCIS   | change_attribute | R@2     | 22.36         | **34.91** | +12.55 | +56.1% |
| GeneCIS   | change_attribute | R@3     | 31.74         | **45.76** | +14.02 | +44.2% |
| GeneCIS   | focus_attribute  | R@1     | 18.82         | **27.83** | +9.01  | +47.9% |
| GeneCIS   | focus_attribute  | R@2     | 30.88         | **42.99** | +12.11 | +39.2% |
| GeneCIS   | focus_attribute  | R@3     | 41.24         | **54.95** | +13.71 | +33.2% |


## 与原论文 OSrCIR 对比

> 原论文配置: OSrCIR (GPT-4o + CLIP ViT-L/14)；当前结果: Qwen-VL-Max + CLIP ViT-L/14
> 可比性说明: FashionIQ 为 `val vs val` 可直接比较；GeneCIS 基本可比；CIRR / CIRCO 因 `paper=test, ours=val` 仅作参考。


| 数据集       | 子集               | 指标      | 原论文 OSrCIR | 三路融合  | 对论文差值  | 可比性   |
| --------- | ---------------- | ------- | ---------- | ----- | ------ | ----- |
| FashionIQ | dress            | R@10    | 29.70      | 19.29 | -10.41 | 可直接比较 |
| FashionIQ | dress            | R@50    | 51.81      | 38.16 | -13.65 | 可直接比较 |
| FashionIQ | shirt            | R@10    | 33.17      | 27.35 | -5.82  | 可直接比较 |
| FashionIQ | shirt            | R@50    | 52.03      | 44.84 | -7.19  | 可直接比较 |
| FashionIQ | toptee           | R@10    | 36.92      | 27.35 | -9.57  | 可直接比较 |
| FashionIQ | toptee           | R@50    | 59.27      | 46.44 | -12.83 | 可直接比较 |
| CIRCO     | —                | mAP@5   | 23.87      | 20.36 | -3.51  | 仅参考   |
| CIRCO     | —                | mAP@10  | 25.33      | 21.26 | -4.07  | 仅参考   |
| CIRCO     | —                | mAP@25  | 27.84      | 23.16 | -4.68  | 仅参考   |
| CIRCO     | —                | mAP@50  | 28.97      | 24.06 | -4.91  | 仅参考   |
| CIRR      | —                | R@1     | 29.45      | 25.90 | -3.55  | 仅参考   |
| CIRR      | —                | R@5     | 57.68      | 56.28 | -1.40  | 仅参考   |
| CIRR      | —                | R@10    | 69.86      | 68.72 | -1.14  | 仅参考   |
| CIRR      | —                | R_sub@1 | 62.12      | 48.89 | -13.23 | 仅参考   |
| CIRR      | —                | R_sub@2 | 81.92      | 70.06 | -11.86 | 仅参考   |
| CIRR      | —                | R_sub@3 | 91.10      | 82.40 | -8.70  | 仅参考   |
| GeneCIS   | change_object    | R@1     | 18.40      | 25.51 | +7.11  | 基本可比  |
| GeneCIS   | change_object    | R@2     | 30.60      | 40.26 | +9.66  | 基本可比  |
| GeneCIS   | change_object    | R@3     | 38.30      | 50.77 | +12.47 | 基本可比  |
| GeneCIS   | focus_object     | R@1     | 15.00      | 23.62 | +8.62  | 基本可比  |
| GeneCIS   | focus_object     | R@2     | 23.60      | 37.04 | +13.44 | 基本可比  |
| GeneCIS   | focus_object     | R@3     | 34.20      | 47.45 | +13.25 | 基本可比  |
| GeneCIS   | change_attribute | R@1     | 17.20      | 21.79 | +4.59  | 基本可比  |
| GeneCIS   | change_attribute | R@2     | 28.50      | 34.91 | +6.41  | 基本可比  |
| GeneCIS   | change_attribute | R@3     | 37.90      | 45.76 | +7.86  | 基本可比  |
| GeneCIS   | focus_attribute  | R@1     | 20.90      | 27.83 | +6.93  | 基本可比  |
| GeneCIS   | focus_attribute  | R@2     | 33.10      | 42.99 | +9.89  | 基本可比  |
| GeneCIS   | focus_attribute  | R@3     | 44.50      | 54.95 | +10.45 | 基本可比  |


> 对论文结论: FashionIQ 当前低于原论文已报告的 6 个指标；GeneCIS 12/12 指标均超过原论文；CIRR / CIRCO 当前也低于论文数字，但由于评测 split 不同，仅能作为参考。

> **35/35 指标全部提升，0 下降。** FIQ/CIRCO/CIRR 相对提升 +2%~~+35%；GeneCIS 相对提升 +33%~~+85%。

