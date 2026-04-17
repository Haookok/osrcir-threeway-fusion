"""
生成本科毕业答辩 PPT (v2)
- 16:9 宽屏，总 16 页
- 学术风格：蓝色系 + 白色背景，简洁大气
- 左图右文 / 左文右图 / 全图 三种布局混用
- 图片来自 docs/thesis/figures/*.png
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(ROOT, '..', 'thesis', 'figures'))
OUT = os.path.join(ROOT, 'defense.pptx')

TONGJI_BLUE = RGBColor(0x00, 0x4A, 0x8F)
DARK_BLUE = RGBColor(0x1E, 0x3A, 0x8A)
ACCENT_RED = RGBColor(0xDC, 0x26, 0x26)
ACCENT_GREEN = RGBColor(0x16, 0xA3, 0x4A)
GRAY = RGBColor(0x4B, 0x55, 0x63)
LIGHT_GRAY = RGBColor(0xE5, 0xE7, 0xEB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG_LIGHT = RGBColor(0xF8, 0xFA, 0xFC)

FONT_CN = 'Microsoft YaHei'
FONT_EN = 'Calibri'

SLIDE_W = 13.333
SLIDE_H = 7.5


def add_text(slide, left, top, width, height, text, size=18, bold=False,
             color=None, align=PP_ALIGN.LEFT, font=FONT_CN, anchor=MSO_ANCHOR.TOP,
             line_space=None):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.clear()
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line_space is not None:
            p.line_spacing = line_space
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color
    return tb


def add_bullets(slide, left, top, width, height, bullets, size=16, color=GRAY,
                line_space=1.3):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_space
        p.space_before = Pt(4)
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = '▸ ' + text
        run.font.name = FONT_CN
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return tb


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
    shape.shadow.inherit = False
    return shape


def add_page_header(slide, page_num, total, chapter=''):
    add_rect(slide, 0, 0, SLIDE_W, 0.08, TONGJI_BLUE)
    add_rect(slide, 0, SLIDE_H - 0.08, SLIDE_W, 0.08, TONGJI_BLUE)
    if chapter:
        add_text(slide, 0.5, 0.2, 5, 0.3, chapter, size=11, color=GRAY)
    add_text(slide, SLIDE_W - 1.5, SLIDE_H - 0.5, 1.2, 0.3,
             f'{page_num} / {total}', size=11, color=GRAY,
             align=PP_ALIGN.RIGHT, font=FONT_EN)


def add_title_bar(slide, title, subtitle=''):
    add_rect(slide, 0.5, 0.75, 0.12, 0.7, TONGJI_BLUE)
    add_text(slide, 0.75, 0.7, 10, 0.55, title,
             size=26, bold=True, color=DARK_BLUE, anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_text(slide, 0.75, 1.25, 10, 0.35, subtitle,
                 size=13, color=GRAY, font=FONT_EN)
    add_rect(slide, 0.5, 1.65, SLIDE_W - 1.0, 0.02, LIGHT_GRAY)


prs = Presentation()
prs.slide_width = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)
BLANK = prs.slide_layouts[6]
TOTAL_PAGES = 16
page_idx = 0


def new_slide():
    global page_idx
    page_idx += 1
    return prs.slides.add_slide(BLANK)


# ============= PAGE 1 封面 =============
s = new_slide()
add_rect(s, 0, 0, SLIDE_W, 2.2, TONGJI_BLUE)
add_text(s, 0, 0.5, SLIDE_W, 0.5, 'TONGJI UNIVERSITY',
         size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER, font=FONT_EN)
add_text(s, 0, 1.0, SLIDE_W, 0.6, '同济大学  本科毕业设计（论文）答辩',
         size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, 1, 2.8, 11.3, 1.0, '基于视觉代理与描述融合的',
         size=38, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
add_text(s, 1, 3.8, 11.3, 1.0, '零样本组合式图像检索改进',
         size=38, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
add_rect(s, 4.5, 5.0, 4.3, 0.04, TONGJI_BLUE)
add_text(s, 0, 5.3, SLIDE_W, 0.4,
         '学院：电子与信息工程学院    专业：计算机科学与技术',
         size=18, color=GRAY, align=PP_ALIGN.CENTER)
add_text(s, 0, 5.8, SLIDE_W, 0.4,
         '答辩人：杨昊明    指导教师：（填写指导教师）',
         size=18, color=GRAY, align=PP_ALIGN.CENTER)
add_text(s, 0, 6.6, SLIDE_W, 0.4, '2026 年 5 月',
         size=14, color=GRAY, align=PP_ALIGN.CENTER, font=FONT_EN)
add_rect(s, 0, SLIDE_H - 0.3, SLIDE_W, 0.3, TONGJI_BLUE)

# ============= PAGE 2 目录 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '目录 / Contents')
add_title_bar(s, '目录', 'Contents')
contents = [
    ('01', '研究背景与意义', 'Background & Motivation'),
    ('02', '相关工作与研究问题', 'Related Work & Problems'),
    ('03', '三路融合方法设计', 'Three-Way Fusion Method'),
    ('04', '实验结果与分析', 'Experiments & Results'),
    ('05', '总结与展望', 'Conclusion & Future Work'),
]
y = 2.2
for num, cn, en in contents:
    circle = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.5), Inches(y),
                                Inches(0.75), Inches(0.75))
    circle.fill.solid(); circle.fill.fore_color.rgb = TONGJI_BLUE
    circle.line.fill.background()
    tf = circle.text_frame
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = num; r.font.size = Pt(20); r.font.bold = True
    r.font.color.rgb = WHITE; r.font.name = FONT_EN
    add_text(s, 2.5, y + 0.05, 8, 0.4, cn, size=20, bold=True, color=DARK_BLUE)
    add_text(s, 2.5, y + 0.5, 8, 0.3, en, size=12, color=GRAY, font=FONT_EN)
    y += 0.95

# ============= PAGE 3 研究背景 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '01  研究背景与意义')
add_title_bar(s, '研究背景与意义', 'Background & Motivation')
add_text(s, 0.6, 2.0, 6.5, 0.4, '什么是组合式图像检索 (CIR)',
         size=18, bold=True, color=TONGJI_BLUE)
add_bullets(s, 0.6, 2.5, 6.5, 3.0, [
    '输入：一张参考图像 + 一段修改文本',
    '输出：图库中满足"保留关键内容 + 应用修改"的目标图像',
    '例：同款连衣裙改为红色 / 将猫替换为狗',
    '零样本 (ZS-CIR)：不在目标数据集上训练，更贴近实际应用',
], size=15, line_space=1.5)
add_text(s, 0.6, 5.3, 6.5, 0.4, '为什么选择这个课题', size=18, bold=True, color=TONGJI_BLUE)
add_bullets(s, 0.6, 5.8, 6.5, 1.5, [
    '多模态大模型快速发展，给 ZS-CIR 带来新机会',
    'OSrCIR (CVPR 2025 Highlight) 为该方向代表工作',
    '实际应用价值大：电商搜索 / 内容创作 / 图像编辑',
], size=13, line_space=1.3)
add_rect(s, 7.8, 2.0, 5.0, 5.0, BG_LIGHT)
add_text(s, 7.8, 2.1, 5.0, 0.4, '任务示意图',
         size=14, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
add_rect(s, 8.2, 2.8, 1.3, 1.3, RGBColor(0xDB, 0xEA, 0xFE), TONGJI_BLUE)
add_text(s, 8.2, 2.8, 1.3, 1.3, '参考图', size=11, color=DARK_BLUE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_text(s, 8.2, 4.15, 1.3, 0.3, '(Reference)', size=9, color=GRAY,
         align=PP_ALIGN.CENTER, font=FONT_EN)
add_text(s, 9.5, 3.0, 0.4, 0.9, '+', size=32, bold=True, color=TONGJI_BLUE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_rect(s, 9.9, 2.8, 1.8, 1.3, RGBColor(0xFE, 0xF3, 0xC7), RGBColor(0xCA, 0x8A, 0x04))
add_text(s, 9.9, 2.85, 1.8, 1.3, '"change\ncolor to red"',
         size=11, color=GRAY, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, font=FONT_EN)
add_text(s, 9.9, 4.15, 1.8, 0.3, '(Modification)', size=9, color=GRAY,
         align=PP_ALIGN.CENTER, font=FONT_EN)
add_text(s, 11.7, 3.0, 0.7, 0.9, '→', size=32, bold=True, color=TONGJI_BLUE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=FONT_EN)
add_rect(s, 10.7, 4.7, 1.8, 1.8, RGBColor(0xFE, 0xE2, 0xE2), ACCENT_RED)
add_text(s, 10.7, 4.7, 1.8, 1.8, '目标图', size=13, bold=True, color=ACCENT_RED,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_text(s, 10.7, 6.5, 1.8, 0.3, '(Target)', size=9, color=GRAY,
         align=PP_ALIGN.CENTER, font=FONT_EN)
add_text(s, 8.0, 4.6, 2.5, 0.4, '输入 (Query)', size=11, color=GRAY,
         align=PP_ALIGN.CENTER, bold=True)

# ============= PAGE 4 相关工作与问题 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '02  相关工作与研究问题')
add_title_bar(s, '相关工作：两条技术路线', 'Related Work')
add_rect(s, 0.6, 2.0, 6.0, 2.2, BG_LIGHT)
add_text(s, 0.8, 2.15, 5.6, 0.4, '① 基于映射的方法',
         size=18, bold=True, color=TONGJI_BLUE)
add_bullets(s, 0.8, 2.7, 5.6, 1.5, [
    '代表：Pic2Word, SEARLE, LinCIR',
    '思路：把图像映射为文本伪词后组合',
    '优点：推理简单、效率高',
    '缺点：视觉信息压缩，细粒度丢失',
], size=12, line_space=1.2)
add_rect(s, 6.8, 2.0, 6.0, 2.2, BG_LIGHT)
add_text(s, 7.0, 2.15, 5.6, 0.4, '② 基于推理的方法 ★',
         size=18, bold=True, color=ACCENT_RED)
add_bullets(s, 7.0, 2.7, 5.6, 1.5, [
    '代表：CIReVL, OSrCIR (本文基线)',
    '思路：MLLM 直接推理出目标描述',
    '优点：多模态推理能力强，无需训练',
    '缺点：单次推理、缺乏视觉校验',
], size=12, line_space=1.2)
add_text(s, 0.6, 4.6, 12, 0.5, '现有方法存在的三大问题',
         size=18, bold=True, color=DARK_BLUE)
problems = [
    ('⚠ 问题 1', '初始描述缺乏视觉验证', '一次性生成，偏差无法纠正'),
    ('⚠ 问题 2', '单一路径检索脆弱', 'CLIP 对文本敏感，单点误差放大'),
    ('⚠ 问题 3', 'AI 生成图易引入幻觉', '直接描述代理图反而加重误差'),
]
for i, (tag, title, desc) in enumerate(problems):
    x = 0.6 + i * 4.1
    add_rect(s, x, 5.2, 3.9, 1.7, WHITE, ACCENT_RED)
    add_text(s, x + 0.15, 5.3, 3.6, 0.35, tag, size=12, bold=True, color=ACCENT_RED)
    add_text(s, x + 0.15, 5.65, 3.6, 0.45, title, size=15, bold=True, color=DARK_BLUE)
    add_text(s, x + 0.15, 6.15, 3.6, 0.7, desc, size=11, color=GRAY, line_space=1.3)

# ============= PAGE 5 三层创新 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '02  相关工作与研究问题')
add_title_bar(s, '本文主要工作：三层创新', 'Three-Layer Innovation')
add_text(s, 0.6, 1.9, 12, 0.4,
         '针对上述三大问题，本文提出 "三路融合" (Three-Way Fusion) 改进框架',
         size=16, color=GRAY)
cards = [
    {'num': '01', 'color': TONGJI_BLUE, 'title': '视觉代理机制', 'en': 'Visual Proxy',
     'desc': '将初始描述 D₁ 输入文生图模型，生成代理图像，\n为检索引入图像空间辅助信号',
     'solve': '解决问题 ①'},
    {'num': '02', 'color': ACCENT_GREEN, 'title': '反幻觉精炼策略', 'en': 'Anti-Hallucination CoT',
     'desc': 'V7 Prompt 限定代理图仅作"诊断工具"，\n切断幻觉细节写入描述的路径',
     'solve': '解决问题 ③'},
    {'num': '03', 'color': ACCENT_RED, 'title': '描述融合 + 三路融合', 'en': 'Description Ensemble',
     'desc': '在 CLIP 特征空间加权融合 D₁/D₂/代理图，\n三路信号互补，稳健性强',
     'solve': '解决问题 ②'},
]
for i, c in enumerate(cards):
    x = 0.6 + i * 4.1
    add_rect(s, x, 2.5, 3.9, 4.2, WHITE, c['color'])
    add_rect(s, x, 2.5, 3.9, 0.6, c['color'])
    add_text(s, x, 2.55, 3.9, 0.55, c['num'],
             size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER, font=FONT_EN)
    add_text(s, x + 0.2, 3.25, 3.5, 0.5, c['title'],
             size=17, bold=True, color=DARK_BLUE)
    add_text(s, x + 0.2, 3.75, 3.5, 0.3, c['en'],
             size=11, color=GRAY, font=FONT_EN)
    add_text(s, x + 0.2, 4.25, 3.5, 1.5, c['desc'],
             size=12, color=GRAY, line_space=1.4)
    add_rect(s, x + 0.2, 6.0, 3.5, 0.5, c['color'])
    add_text(s, x + 0.2, 6.05, 3.5, 0.4, c['solve'],
             size=12, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ============= PAGE 6 总体框架 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '03  三路融合方法设计')
add_title_bar(s, '方法总体框架', 'Overall Pipeline')
pipeline_png = os.path.join(FIG_DIR, 'fig_pipeline.png')
if os.path.exists(pipeline_png):
    s.shapes.add_picture(pipeline_png, Inches(0.6), Inches(1.9), width=Inches(9.0))
add_rect(s, 9.9, 1.9, 3.1, 5.0, BG_LIGHT)
add_text(s, 9.9, 2.0, 3.1, 0.4, '核心公式',
         size=14, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
add_rect(s, 10.0, 2.5, 2.9, 0.02, TONGJI_BLUE)
add_text(s, 10.0, 2.6, 2.9, 0.3, '文本融合 (β=0.7)',
         size=11, bold=True, color=TONGJI_BLUE)
add_text(s, 10.0, 2.95, 2.9, 0.8,
         'f_text = normalize(\n  β·CLIP(D₁) +\n  (1-β)·CLIP(D₂))',
         size=10, color=GRAY, font=FONT_EN, line_space=1.2)
add_text(s, 10.0, 3.9, 2.9, 0.3, '三路融合 (α=0.9)',
         size=11, bold=True, color=ACCENT_RED)
add_text(s, 10.0, 4.25, 2.9, 0.8,
         'score =\n  α·sim(f_text, g) +\n  (1-α)·sim(CLIP(P), g)',
         size=10, color=GRAY, font=FONT_EN, line_space=1.2)
add_text(s, 10.0, 5.3, 2.9, 0.3, '参数含义',
         size=11, bold=True, color=DARK_BLUE)
add_bullets(s, 10.0, 5.6, 2.9, 1.3, [
    'β: D₁ 与 D₂ 的权重',
    'α: 文本与代理图权重',
], size=10, line_space=1.3)

# ============= PAGE 7 视觉代理 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '03  三路融合方法设计')
add_title_bar(s, '创新点 1：视觉代理机制', 'Visual Proxy')
add_text(s, 0.6, 2.0, 6.0, 0.4, '设计动机', size=18, bold=True, color=TONGJI_BLUE)
add_bullets(s, 0.6, 2.5, 6.0, 2.2, [
    'Baseline 的 D₁ 一旦有误，无任何信号帮助纠正',
    '把 D₁ 投射到图像空间，获得可视化的中间参照',
    '代理图不是目标图，但具有检索价值',
    '两个作用：① 提供图像特征  ② 辅助下一轮精炼',
], size=14, line_space=1.4)
add_text(s, 0.6, 4.7, 6.0, 0.4, '实现细节', size=18, bold=True, color=TONGJI_BLUE)
add_bullets(s, 0.6, 5.2, 6.0, 1.8, [
    '文生图模型：MiniMax image-01',
    '输入：D₁ 初始描述文本',
    '输出：512×512 代理图像',
    '成本：约 0.03 元 / 张',
], size=13, line_space=1.3)
add_rect(s, 7.2, 2.0, 5.6, 5.0, BG_LIGHT)
add_text(s, 7.2, 2.1, 5.6, 0.4, '初步验证 (FashionIQ dress, 50 样本)',
         size=14, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
methods = [
    ('Baseline\n(仅 D₁)', 18.0, LIGHT_GRAY, GRAY),
    ('Plan A\n后融合', 26.0, RGBColor(0xBB, 0xF7, 0xD0), ACCENT_GREEN),
    ('Plan B\n前融合', 22.0, RGBColor(0xBF, 0xDB, 0xFE), TONGJI_BLUE),
]
max_val = 30
chart_base_y = 6.3
chart_h = 3.0
for i, (name, val, fill, edge) in enumerate(methods):
    x = 7.6 + i * 1.75
    h = val / max_val * chart_h
    add_rect(s, x, chart_base_y - h, 1.3, h, fill, edge)
    add_text(s, x - 0.1, chart_base_y - h - 0.4, 1.5, 0.3, f'{val}',
             size=13, bold=True, color=edge, align=PP_ALIGN.CENTER, font=FONT_EN)
    add_text(s, x - 0.1, chart_base_y + 0.05, 1.5, 0.6, name,
             size=10, color=GRAY, align=PP_ALIGN.CENTER, line_space=1.1)
add_text(s, 7.2, 2.6, 5.6, 0.4, 'R@10 (%)',
         size=11, color=GRAY, align=PP_ALIGN.CENTER, font=FONT_EN)
add_rect(s, 7.4, 6.5, 5.2, 0.45, RGBColor(0xDC, 0xFC, 0xE7))
add_text(s, 7.4, 6.55, 5.2, 0.35,
         '✓ 代理图有效：+8 pp (后融合) / +4 pp (前融合)',
         size=11, bold=True, color=ACCENT_GREEN, align=PP_ALIGN.CENTER)

# ============= PAGE 8 反幻觉 V7 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '03  三路融合方法设计')
add_title_bar(s, '创新点 2：反幻觉精炼 (V7 Prompt)', 'Anti-Hallucination CoT')
prompt_png = os.path.join(FIG_DIR, 'fig_prompt_evolution.png')
if os.path.exists(prompt_png):
    s.shapes.add_picture(prompt_png, Inches(0.5), Inches(1.95), width=Inches(6.5))
add_text(s, 7.4, 2.0, 5.5, 0.4, 'V7 核心思想', size=18, bold=True, color=TONGJI_BLUE)
add_text(s, 7.4, 2.5, 5.5, 0.4, '"代理图只做诊断，不做描述来源"',
         size=15, bold=True, color=ACCENT_RED)
add_text(s, 7.4, 3.1, 5.5, 0.35, '三条关键约束：',
         size=13, bold=True, color=DARK_BLUE)
add_bullets(s, 7.4, 3.5, 5.5, 2.2, [
    '声明代理图是 AI 生成，可能含幻觉',
    '仅用代理图检查修改是否正确应用',
    '禁止复制代理图中的视觉细节',
    '输出描述尽量简短',
], size=12, line_space=1.3)
add_rect(s, 7.4, 5.7, 5.5, 1.3, RGBColor(0xFE, 0xE2, 0xE2))
add_text(s, 7.55, 5.8, 5.3, 0.35, '⚠ 关键发现', size=12, bold=True, color=ACCENT_RED)
add_text(s, 7.55, 6.15, 5.3, 0.85,
         'V6 加入更多上下文反而灾难性退化\n'
         '(CIRR R@10 -17.5)，说明限制信息来源\n'
         '比增加信息更有效',
         size=10, color=GRAY, line_space=1.3)

# ============= PAGE 9 描述融合 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '03  三路融合方法设计')
add_title_bar(s, '创新点 3：描述融合策略', 'Description Ensemble')
add_text(s, 0.6, 2.0, 6.0, 0.4, '为什么用 "融合" 而非 "替换"',
         size=16, bold=True, color=TONGJI_BLUE)
add_bullets(s, 0.6, 2.5, 6.0, 3.0, [
    'V7 精炼 D₂ 在多数样本上有益，但部分样本信息压缩',
    '直接用 D₂ 替换 D₁ = 全有或全无的决策',
    '在 CLIP 特征空间加权平均 = 软决策',
    'D₂ 表现好 → 融合有增益',
    'D₂ 表现差 → D₁ 主导不退化',
], size=13, line_space=1.4)
add_rect(s, 7.0, 2.0, 5.8, 5.2, BG_LIGHT)
add_text(s, 7.0, 2.1, 5.8, 0.4, '效果对比 (CIRR R@10 vs shirt R@10)',
         size=13, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
rows = [
    ('方法', 'CIRR R@10', 'shirt R@10', True, LIGHT_GRAY),
    ('Baseline (仅 D₁)', '67.0', '30.0', False, WHITE),
    ('V7 替换 (仅 D₂)', '73.5 (+6.5)', '26.0 (-4.0)', False, RGBColor(0xFE, 0xE2, 0xE2)),
    ('V7 + Ensemble', '69.0 (+2.0) ✓', '30.5 (+0.5) ✓', False, RGBColor(0xDC, 0xFC, 0xE7)),
]
table_y = 2.7
for i, (c1, c2, c3, hdr, bg) in enumerate(rows):
    y = table_y + i * 0.55
    add_rect(s, 7.2, y, 5.4, 0.55, bg, GRAY if hdr else None)
    add_text(s, 7.25, y, 2.4, 0.55, c1, size=11, bold=hdr,
             color=DARK_BLUE if hdr else GRAY, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, 9.65, y, 1.5, 0.55, c2, size=11, bold=hdr, color=GRAY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=FONT_EN)
    add_text(s, 11.15, y, 1.45, 0.55, c3, size=11, bold=hdr, color=GRAY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=FONT_EN)
add_rect(s, 7.2, 5.3, 5.4, 1.6, WHITE, TONGJI_BLUE)
add_text(s, 7.3, 5.4, 5.2, 0.35, '关键结论', size=12, bold=True, color=TONGJI_BLUE)
add_text(s, 7.3, 5.8, 5.2, 1.0,
         '融合 (β=0.7) 牺牲 V7 在 CIRR 上的极端提升，\n'
         '但换来 shirt 等原本退化的子集的正向提升，\n'
         '最终实现 "所有数据集都不退化" 的稳健性',
         size=10, color=GRAY, line_space=1.4)

# ============= PAGE 10 实验设置 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '04  实验结果与分析')
add_title_bar(s, '实验设置', 'Experimental Setup')
add_text(s, 0.6, 2.0, 6.0, 0.4, '① 评估数据集 (4 基准 / 9 子任务)',
         size=16, bold=True, color=TONGJI_BLUE)
ds_rows = [
    ('FashionIQ', '服饰检索', '3 子集', '~6K'),
    ('CIRCO', '开放域', 'val split', '123K'),
    ('CIRR', '自然场景', '4181 查询', '2297'),
    ('GeneCIS', '细粒度', '4 子任务', '~14'),
]
y = 2.5
add_rect(s, 0.6, y, 6.0, 0.4, LIGHT_GRAY)
for col, txt in zip([0.7, 2.0, 3.5, 4.8], ['数据集', '类型', '规模', 'Gallery']):
    add_text(s, col, y, 1.5, 0.4, txt, size=11, bold=True,
             color=DARK_BLUE, anchor=MSO_ANCHOR.MIDDLE)
for i, (ds, tp, scale, g) in enumerate(ds_rows):
    y2 = y + 0.4 + i * 0.45
    bg = BG_LIGHT if i % 2 == 0 else WHITE
    add_rect(s, 0.6, y2, 6.0, 0.45, bg)
    for col, txt in zip([0.7, 2.0, 3.5, 4.8], [ds, tp, scale, g]):
        add_text(s, col, y2, 1.5, 0.45, txt, size=11, color=GRAY,
                 anchor=MSO_ANCHOR.MIDDLE)
add_text(s, 7.0, 2.0, 6.0, 0.4, '② 实现细节',
         size=16, bold=True, color=TONGJI_BLUE)
add_bullets(s, 7.0, 2.5, 6.0, 3.5, [
    'MLLM：Qwen-VL-Max (阿里云 DashScope)',
    '文生图：MiniMax image-01',
    'CLIP：ViT-L/14（与论文一致）',
    '默认参数：β=0.7, α=0.9',
    'GeneCIS：专用 prompt + 任务自适应 α/β',
], size=13, line_space=1.5)
add_rect(s, 0.6, 5.0, 12.4, 2.0, RGBColor(0xFE, 0xF3, 0xC7))
add_text(s, 0.8, 5.15, 12, 0.4, '⚠ 关于基线复现的说明',
         size=14, bold=True, color=RGBColor(0xCA, 0x8A, 0x04))
add_text(s, 0.8, 5.55, 12, 1.35,
         '原论文 OSrCIR 使用 GPT-4o，本工作因 API 可用性限制替换为 Qwen-VL-Max。\n'
         '因此本工作 Baseline 的绝对数值在部分数据集上低于原论文，\n'
         '但所有改进实验均基于同一 Baseline，不同方案之间的对比仍然公平。',
         size=12, color=GRAY, line_space=1.4)

# ============= PAGE 11 主要结果 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '04  实验结果与分析')
add_title_bar(s, '主要结果：9 数据集全面提升', 'Main Results on 9 Benchmarks')
main_png = os.path.join(FIG_DIR, 'fig_main_results.png')
if os.path.exists(main_png):
    s.shapes.add_picture(main_png, Inches(0.6), Inches(1.95), width=Inches(12.2))
stats = [
    ('9/9', '数据集主指标提升', TONGJI_BLUE),
    ('35/35', '评估指标全部正向', ACCENT_GREEN),
    ('+31.2%', 'CIRCO mAP@10 最大相对提升', ACCENT_RED),
    ('+84.5%', 'GeneCIS ch_obj 最大提升', DARK_BLUE),
]
for i, (num, desc, color) in enumerate(stats):
    x = 0.6 + i * 3.15
    add_rect(s, x, 5.8, 3.0, 1.25, WHITE, color)
    add_text(s, x, 5.9, 3.0, 0.55, num,
             size=22, bold=True, color=color,
             align=PP_ALIGN.CENTER, font=FONT_EN)
    add_text(s, x, 6.45, 3.0, 0.55, desc,
             size=11, color=GRAY, align=PP_ALIGN.CENTER)

# ============= PAGE 12 Prompt 消融 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '04  实验结果与分析')
add_title_bar(s, '消融实验 (1)：Prompt 版本演进', 'Ablation: Prompt Evolution')
if os.path.exists(prompt_png):
    s.shapes.add_picture(prompt_png, Inches(0.5), Inches(1.95), width=Inches(7.0))
add_text(s, 7.8, 2.0, 5.0, 0.4, '关键洞察', size=16, bold=True, color=TONGJI_BLUE)
findings = [
    ('❌ Original / V5', '代理图细节被复制到描述', ACCENT_RED),
    ('💀 V6 (全上下文)', '信息过载，灾难性崩溃', ACCENT_RED),
    ('✅ V7 突破点', '代理图仅做诊断，禁复制', ACCENT_GREEN),
    ('⚖ V7 + Ensemble', '牺牲极端换稳健', TONGJI_BLUE),
]
y = 2.55
for tag, desc, color in findings:
    add_rect(s, 7.8, y, 5.0, 0.85, WHITE, color)
    add_text(s, 7.95, y + 0.08, 4.8, 0.35, tag, size=12, bold=True, color=color)
    add_text(s, 7.95, y + 0.42, 4.8, 0.35, desc, size=11, color=GRAY)
    y += 1.0

# ============= PAGE 13 α/β 参数 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '04  实验结果与分析')
add_title_bar(s, '消融实验 (2)：α/β 参数敏感性', 'Ablation: α/β Grid Search')
heatmap_png = os.path.join(FIG_DIR, 'fig_heatmap.png')
if os.path.exists(heatmap_png):
    s.shapes.add_picture(heatmap_png, Inches(0.4), Inches(1.95), width=Inches(8.5))
add_text(s, 9.1, 2.0, 3.9, 0.4, '各子集最优参数',
         size=14, bold=True, color=TONGJI_BLUE)
grid_rows = [
    ('子集', 'β', 'α', 'ΔR@1', True),
    ('change_object', '0.30', '0.95', '+1.02', False),
    ('focus_object', '1.00', '0.90', '+0.05', False),
    ('change_attr.', '0.80', '0.80', '+1.33', False),
    ('focus_attr.', '0.60', '0.85', '+1.55', False),
]
y_t = 2.5
for i, (c1, c2, c3, c4, hdr) in enumerate(grid_rows):
    y = y_t + i * 0.45
    bg = LIGHT_GRAY if hdr else (BG_LIGHT if i % 2 == 1 else WHITE)
    add_rect(s, 9.1, y, 3.9, 0.45, bg)
    add_text(s, 9.15, y, 1.6, 0.45, c1, size=10, bold=hdr,
             color=DARK_BLUE if hdr else GRAY, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, 10.75, y, 0.7, 0.45, c2, size=10, bold=hdr, color=GRAY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=FONT_EN)
    add_text(s, 11.45, y, 0.7, 0.45, c3, size=10, bold=hdr, color=GRAY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=FONT_EN)
    add_text(s, 12.15, y, 0.8, 0.45, c4, size=10, bold=hdr,
             color=ACCENT_GREEN if not hdr else DARK_BLUE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=FONT_EN)
add_text(s, 9.1, 5.0, 3.9, 0.4, '关键发现', size=14, bold=True, color=TONGJI_BLUE)
add_bullets(s, 9.1, 5.45, 3.9, 2.0, [
    '各子集最优参数差异大',
    'focus_object β=1 最优',
    '证明 task-adaptive 的价值',
], size=10, line_space=1.4)

# ============= PAGE 14 贡献总结 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '05  总结与展望')
add_title_bar(s, '主要贡献总结', 'Summary of Contributions')
add_text(s, 0.6, 2.0, 12, 0.4, '本文的主要创新与贡献',
         size=18, bold=True, color=TONGJI_BLUE)
contribs = [
    ('🎯 方法创新',
     '提出首个面向 ZS-CIR 的三路融合框架，同时引入视觉代理、\n反幻觉精炼和描述融合三个新模块'),
    ('📊 实验完整',
     '在 4 基准 9 子任务上完成全量评估，35 项指标全部正向提升，\n无持平、无退化'),
    ('🔬 工程贡献',
     '完整复现并开源 OSrCIR 基线，基于 Qwen-VL-Max 重构 MLLM 流程，\n累计花费约 140 元完成全流程实验'),
    ('💡 理论洞察',
     '证明 "限制信息来源比增加信息更有效" 这一反直觉结论，\n以及 task-adaptive 参数配置对细粒度任务的必要性'),
]
y = 2.55
for tag, desc in contribs:
    add_rect(s, 0.6, y, 12.4, 1.0, WHITE, TONGJI_BLUE)
    add_text(s, 0.8, y + 0.1, 2.5, 0.8, tag,
             size=14, bold=True, color=TONGJI_BLUE, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, 3.3, y + 0.1, 9.5, 0.8, desc,
             size=12, color=GRAY, line_space=1.4, anchor=MSO_ANCHOR.MIDDLE)
    y += 1.1

# ============= PAGE 15 局限与展望 =============
s = new_slide()
add_page_header(s, page_idx, TOTAL_PAGES, '05  总结与展望')
add_title_bar(s, '局限与未来工作', 'Limitations & Future Work')
add_rect(s, 0.6, 2.0, 6.0, 5.0, BG_LIGHT)
add_text(s, 0.8, 2.15, 5.6, 0.5, '本文方法的局限',
         size=16, bold=True, color=ACCENT_RED)
add_bullets(s, 0.8, 2.7, 5.6, 3.0, [
    '相比基线多一次文生图 + 一次精炼推理，成本更高',
    '参数对任务类型敏感，统一默认不是全局最优',
    '方法性能上限仍受 MLLM 能力影响',
    'Qwen-VL-Max 导致 Baseline 数值低于原论文',
], size=13, line_space=1.6)
add_rect(s, 6.8, 2.0, 6.0, 5.0, BG_LIGHT)
add_text(s, 7.0, 2.15, 5.6, 0.5, '未来可改进方向',
         size=16, bold=True, color=ACCENT_GREEN)
add_bullets(s, 7.0, 2.7, 5.6, 3.5, [
    '查询自适应参数预测（消除人工调参）',
    '使用更强 MLLM（GPT-4o / Claude）',
    '多代理图策略提升视觉信号质量',
    '扩展到有监督 CIR 任务',
    '代理图质量评估与过滤机制',
], size=13, line_space=1.6)

# ============= PAGE 16 致谢 =============
s = new_slide()
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, WHITE)
add_rect(s, 0, 0, SLIDE_W, 0.2, TONGJI_BLUE)
add_rect(s, 0, SLIDE_H - 0.2, SLIDE_W, 0.2, TONGJI_BLUE)
add_text(s, 0, 2.3, SLIDE_W, 1.0, '谢 谢 聆 听',
         size=60, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
add_text(s, 0, 3.5, SLIDE_W, 0.5, 'Thanks for your attention',
         size=20, color=GRAY, align=PP_ALIGN.CENTER, font=FONT_EN)
add_rect(s, (SLIDE_W - 3) / 2, 4.3, 3, 0.03, TONGJI_BLUE)
add_text(s, 0, 4.6, SLIDE_W, 0.5, '敬请各位老师批评指正',
         size=22, color=GRAY, align=PP_ALIGN.CENTER)
add_text(s, 0, 6.0, SLIDE_W, 0.35,
         '答辩人：杨昊明   |   指导教师：（填写指导教师）',
         size=14, color=GRAY, align=PP_ALIGN.CENTER)
add_text(s, 0, 6.4, SLIDE_W, 0.35,
         '同济大学 电子与信息工程学院   |   2026 年 5 月',
         size=13, color=GRAY, align=PP_ALIGN.CENTER)

prs.save(OUT)
print(f'Saved: {OUT}')
print(f'Total slides: {page_idx}')
