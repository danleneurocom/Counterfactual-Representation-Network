from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAGE_WIDTH = 1224
PAGE_HEIGHT = 1584
MARGIN_X = 96
MARGIN_TOP = 90
MARGIN_BOTTOM = 84
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

TITLE_FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
BODY_FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
BODY_ITALIC_FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"
HEADING_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO_FONT_PATH = "/System/Library/Fonts/Supplemental/Courier New.ttf"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


TITLE_FONT = load_font(TITLE_FONT_PATH, 34)
SUBTITLE_FONT = load_font(BODY_ITALIC_FONT_PATH, 18)
H1_FONT = load_font(HEADING_FONT_PATH, 24)
H2_FONT = load_font(HEADING_FONT_PATH, 19)
BODY_FONT = load_font(BODY_FONT_PATH, 17)
BULLET_FONT = load_font(BODY_FONT_PATH, 17)
CODE_FONT = load_font(MONO_FONT_PATH, 15)
FOOTER_FONT = load_font(BODY_FONT_PATH, 12)


def line_height(font: ImageFont.FreeTypeFont, extra: int = 0) -> int:
    bbox = font.getbbox("Ag")
    return int((bbox[3] - bbox[1]) + extra)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    return float(draw.textlength(text, font=font))


def wrap_words(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def wrap_code_line(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    if text_width(draw, text, font) <= max_width:
        return [text]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = f"{current}{char}"
        if current and text_width(draw, candidate, font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def parse_markdown(text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(("paragraph", " ".join(part.strip() for part in paragraph if part.strip())))
            paragraph = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if line.strip().startswith("```"):
            flush_paragraph()
            if in_code:
                blocks.append(("code", code_lines[:]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            blocks.append(("title", stripped[2:].strip()))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append(("h1", stripped[3:].strip()))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            blocks.append(("h2", stripped[4:].strip()))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", stripped[2:].strip()))
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            blocks.append(("bullet", stripped))
            continue
        paragraph.append(stripped)

    flush_paragraph()
    if code_lines:
        blocks.append(("code", code_lines))
    return blocks


class PdfCanvas:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self._new_page()

    def _new_page(self) -> None:
        self.page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color="white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP
        self.pages.append(self.page)

    def ensure_space(self, height: int) -> None:
        if self.y + height > PAGE_HEIGHT - MARGIN_BOTTOM:
            self._new_page()

    def draw_centered(self, text: str, font: ImageFont.FreeTypeFont, fill: str = "black") -> None:
        bbox = self.draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        x = (PAGE_WIDTH - width) // 2
        self.draw.text((x, self.y), text, font=font, fill=fill)
        self.y += (bbox[3] - bbox[1]) + 10

    def draw_paragraph(self, text: str, font: ImageFont.FreeTypeFont = BODY_FONT, indent: int = 0, spacing: int = 8) -> None:
        lines = wrap_words(self.draw, text, font, CONTENT_WIDTH - indent)
        lh = line_height(font, extra=7)
        self.ensure_space(len(lines) * lh + spacing)
        for line in lines:
            self.draw.text((MARGIN_X + indent, self.y), line, font=font, fill="black")
            self.y += lh
        self.y += spacing

    def draw_bullet(self, text: str) -> None:
        bullet_indent = 28
        wrap_indent = 48
        lines = wrap_words(self.draw, text, BULLET_FONT, CONTENT_WIDTH - wrap_indent)
        lh = line_height(BULLET_FONT, extra=7)
        self.ensure_space(len(lines) * lh + 6)
        self.draw.text((MARGIN_X + bullet_indent - 18, self.y), "-", font=BULLET_FONT, fill="black")
        for index, line in enumerate(lines):
            x = MARGIN_X + wrap_indent if index > 0 else MARGIN_X + bullet_indent
            self.draw.text((x, self.y), line, font=BULLET_FONT, fill="black")
            self.y += lh
        self.y += 6

    def draw_code_block(self, lines: list[str]) -> None:
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(wrap_code_line(self.draw, line, CODE_FONT, CONTENT_WIDTH - 30))
        lh = line_height(CODE_FONT, extra=5)
        block_height = len(wrapped) * lh + 24
        self.ensure_space(block_height + 10)
        x0 = MARGIN_X + 8
        x1 = PAGE_WIDTH - MARGIN_X - 8
        y0 = self.y
        y1 = self.y + block_height
        self.draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill="#f3f4f6", outline="#d1d5db")
        y = y0 + 12
        for line in wrapped:
            self.draw.text((x0 + 12, y), line, font=CODE_FONT, fill="#111827")
            y += lh
        self.y = y1 + 10

    def add_footers(self) -> None:
        total_pages = len(self.pages)
        for index, page in enumerate(self.pages, start=1):
            draw = ImageDraw.Draw(page)
            footer = f"{index}"
            bbox = draw.textbbox((0, 0), footer, font=FOOTER_FONT)
            width = bbox[2] - bbox[0]
            x = (PAGE_WIDTH - width) // 2
            y = PAGE_HEIGHT - MARGIN_BOTTOM + 20
            draw.text((x, y), footer, font=FOOTER_FONT, fill="#555555")
            if total_pages > 1:
                header = "Causal Counterfactual Representation Network"
                draw.text((MARGIN_X, 40), header, font=FOOTER_FONT, fill="#6b7280")

    def save_pdf(self, path: Path) -> None:
        self.add_footers()
        first, *rest = self.pages
        first.save(path, "PDF", resolution=144.0, save_all=True, append_images=rest)


def render_markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    blocks = parse_markdown(input_path.read_text(encoding="utf-8"))
    canvas = PdfCanvas()
    subtitle_drawn = False

    for kind, payload in blocks:
        if kind == "title":
            canvas.draw_centered(str(payload), TITLE_FONT)
            continue
        if not subtitle_drawn:
            canvas.draw_centered("Research Note", SUBTITLE_FONT, fill="#555555")
            canvas.y += 8
            subtitle_drawn = True
        if kind == "h1":
            canvas.ensure_space(line_height(H1_FONT, extra=14) + 8)
            canvas.draw.text((MARGIN_X, canvas.y), str(payload), font=H1_FONT, fill="black")
            canvas.y += line_height(H1_FONT, extra=14)
            continue
        if kind == "h2":
            canvas.ensure_space(line_height(H2_FONT, extra=10) + 4)
            canvas.draw.text((MARGIN_X, canvas.y), str(payload), font=H2_FONT, fill="black")
            canvas.y += line_height(H2_FONT, extra=10)
            continue
        if kind == "paragraph":
            canvas.draw_paragraph(str(payload))
            continue
        if kind == "bullet":
            canvas.draw_bullet(str(payload))
            continue
        if kind == "code":
            canvas.draw_code_block(list(payload))
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save_pdf(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simple academic-style PDF from markdown.")
    parser.add_argument("input", type=Path, help="Input markdown file.")
    parser.add_argument("output", type=Path, help="Output PDF file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_markdown_to_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
