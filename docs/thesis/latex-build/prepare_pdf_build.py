from pathlib import Path
import re


ROOT = Path("/home/haomingyang03/code/osrcir/docs/thesis")
SOURCE = ROOT / "versions" / "v2026-04-14-generic-undergraduate-draft.md"
BUILD_DIR = ROOT / "latex-build"
OUTPUT = BUILD_DIR / "thesis_pdf_source.md"


def skip_front_matter(lines):
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "## 中文摘要":
            start = idx
            break
    if start is None:
        raise RuntimeError("未找到“## 中文摘要”段落，无法生成 PDF 源文件。")
    return lines[start:]


def transform_heading(line: str) -> str:
    stripped = line.strip()

    if stripped == "## 中文摘要":
        return "\\chapter*{摘\\quad 要}\n\\addcontentsline{toc}{chapter}{摘要}\n"
    if stripped == "## Abstract":
        return "\\chapter*{Abstract}\n\\addcontentsline{toc}{chapter}{Abstract}\n"
    if stripped == "## 参考文献":
        return "\\chapter*{参考文献}\n\\addcontentsline{toc}{chapter}{参考文献}\n"
    if stripped == "## 致谢":
        return "\\chapter*{致谢}\n\\addcontentsline{toc}{chapter}{致谢}\n"
    if stripped.startswith("## 附录"):
        title = stripped[3:].strip()
        return f"\\appendix\n# {title}\n"

    match = re.match(r"## 第\d+章\s+(.+)", stripped)
    if match:
        return f"# {match.group(1)}\n"

    match = re.match(r"### \d+\.\d+\s+(.+)", stripped)
    if match:
        return f"## {match.group(1)}\n"

    match = re.match(r"#### \d+\.\d+\.\d+\s+(.+)", stripped)
    if match:
        return f"### {match.group(1)}\n"

    return line


def main():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    body = skip_front_matter(lines)
    transformed = [transform_heading(line) for line in body]

    metadata = """---
title: "基于视觉代理与描述融合的零样本组合式图像检索改进"
author: "通用 LaTeX 模板版初稿"
date: "2026-04-14"
---

"""
    OUTPUT.write_text(metadata + "".join(transformed), encoding="utf-8")


if __name__ == "__main__":
    main()
