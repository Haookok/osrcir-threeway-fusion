"""Convert PROGRESS_REPORT.md to PDF via weasyprint."""
import markdown
import weasyprint
import re

import base64

import os
base_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
md_path = os.path.join(base_dir, 'PROGRESS_REPORT.md')
pdf_path = os.path.join(base_dir, 'PROGRESS_REPORT.pdf')
img_path = os.path.join(base_dir, 'pipeline_diagram.png')

with open(md_path, encoding='utf-8') as f:
    md_text = f.read()

# Embed pipeline diagram as base64
with open(img_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
md_text = md_text.replace(
    '![pipeline](pipeline_diagram.png)',
    f'<img src="data:image/png;base64,{img_b64}" style="width:100%;margin:10px 0;">'
)

html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 2.2cm 2cm 2cm 2cm;
    @bottom-center {{
        content: counter(page);
        font-size: 9pt;
        color: #666;
    }}
}}
body {{
    font-family: "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
                 "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #222;
}}
h1 {{
    font-size: 18pt;
    text-align: center;
    margin-top: 0.5cm;
    margin-bottom: 0.8cm;
    font-weight: 700;
    letter-spacing: 1px;
}}
h2 {{
    font-size: 14pt;
    border-bottom: 1.5px solid #333;
    padding-bottom: 3px;
    margin-top: 1.2em;
    margin-bottom: 0.6em;
    font-weight: 600;
}}
h3 {{
    font-size: 12pt;
    margin-top: 1em;
    margin-bottom: 0.4em;
    font-weight: 600;
}}
p {{
    text-indent: 2em;
    margin: 0.4em 0;
    text-align: justify;
}}
li > p {{
    text-indent: 0;
}}
ul, ol {{
    margin: 0.3em 0;
    padding-left: 2em;
}}
li {{
    margin: 0.15em 0;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.6em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}}
th, td {{
    border: 1px solid #999;
    padding: 4px 6px;
    text-align: center;
}}
th {{
    background: #f0f0f0;
    font-weight: 600;
}}
td:first-child, th:first-child {{
    text-align: left;
}}
code {{
    font-family: "Noto Sans Mono CJK SC", "Source Code Pro", monospace;
    font-size: 9.5pt;
    background: #f5f5f5;
    padding: 1px 3px;
    border-radius: 2px;
}}
strong {{
    font-weight: 700;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

weasyprint.HTML(string=html).write_pdf(pdf_path)
print(f'PDF saved: {pdf_path}')
