# bot.py
# Telegram UI/UX + Copy design-review buddy
# Fix: HTML parse errors in ASCII animations (escape + no < >)

import asyncio
import base64
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import html as py_html
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

CONCEPT_ENABLED = os.getenv("CONCEPT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()

MAX_PREVIEW_BYTES = int(os.getenv("MAX_PREVIEW_BYTES", "8000000"))  # 8 MB

PROGRESS_TOTAL_FRAMES = int(os.getenv("PROGRESS_TOTAL_FRAMES", "28"))
PROGRESS_DELAY = float(os.getenv("PROGRESS_DELAY", "0.12"))

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in environment variables (Railway Variables or local env).")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment variables (Railway Variables or local env).")

client = OpenAI(api_key=OPENAI_API_KEY)
router = Router()


# ----------------------------
# UI
# ----------------------------
BTN_REVIEW = "review"
BTN_HOW = "how"
BTN_PING = "ping"

def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Закинуть на ревью", callback_data=BTN_REVIEW)],
            [InlineKeyboardButton(text="Как это работает?", callback_data=BTN_HOW)],
            [InlineKeyboardButton(text="Ping", callback_data=BTN_PING)],
        ]
    )

WELCOME_TEXT = (
    "Я — партнёр для дизайн-ревью.\n\n"
    "Принимаю на разбор:\n"
    "- скриншоты интерфейса (картинки)\n"
    "- ссылки на Figma фреймы (если файл публичный)\n\n"
    "Жми «Закинуть на ревью» или просто отправь скрин/ссылку."
)

HOW_TEXT = (
    "Как это работает:\n"
    "1) Отправь скриншот или ссылку на публичный Figma фрейм\n"
    "2) Я покажу прогресс обработки\n"
    "3) Верну:\n"
    "   - что я вижу\n"
    "   - вердикт и рекомендации (UX + текст) + оценка\n"
    "   - аннотации на скрине\n"
    "   - концепт, как могло бы быть"
)

PING_TEXT = "pong"


# ----------------------------
# Helpers: safe edit or resend
# ----------------------------
async def safe_edit_or_resend(msg: Message, text: str, parse_mode: Optional[str] = None) -> Message:
    try:
        await msg.edit_text(text, parse_mode=parse_mode)
        return msg
    except Exception:
        try:
            return await msg.answer(text, parse_mode=parse_mode)
        except Exception:
            return msg


def html_escape(s: str) -> str:
    return py_html.escape(s, quote=False)


# ----------------------------
# Retro ASCII animation (dynamic)
# ----------------------------
_SPIN = ["|", "/", "-", "\\"]
_TAPE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
def _ascii_frame(t: int, total: int, title: str) -> str:
    p = max(0.0, min(1.0, t / max(1, total)))
    bar_w = 26
    filled = int(p * bar_w)

    pulse = "@" if (t % 6 in (0, 1)) else "#"
    bar = "[" + (pulse * filled) + ("." * (bar_w - filled)) + "]"
    spin = _SPIN[t % len(_SPIN)]

    box_h = 6
    scan_y = t % box_h

    scroll = t % len(_TAPE)
    ticker = (_TAPE[scroll:] + _TAPE[:scroll])[:22]

    lines = []
    for y in range(box_h):
        if y == scan_y:
            lines.append("|" + ("=" * 28) + "|")
        else:
            noise = "".join(["." if ((x + t + y) % 9) else ":" for x in range(28)])
            lines.append("|" + noise + "|")

    pct = int(p * 100)
    header = f"{title} {spin}  {pct:02d}%"
    # IMPORTANT: no < > here (Telegram HTML parse mode)
    footer = f"{bar}  [{ticker}]"
    return header + "\n" + "\n".join(lines) + "\n" + footer


async def animate_progress(anchor: Message, title: str = "Processing") -> Message:
    frame0 = f"{title}\n{_ascii_frame(0, PROGRESS_TOTAL_FRAMES, title)}"
    msg = await anchor.answer(
        f"{html_escape(title)}\n<code>{html_escape(_ascii_frame(0, PROGRESS_TOTAL_FRAMES, title))}</code>",
        parse_mode=ParseMode.HTML
    )
    for i in range(1, PROGRESS_TOTAL_FRAMES + 1):
        frame = f"{title}\n{_ascii_frame(i, PROGRESS_TOTAL_FRAMES, title)}"
        msg = await safe_edit_or_resend(
            msg,
            f"{html_escape(title)}\n<code>{html_escape(_ascii_frame(i, PROGRESS_TOTAL_FRAMES, title))}</code>",
            parse_mode=ParseMode.HTML,
        )
        await asyncio.sleep(PROGRESS_DELAY)
    return msg


# ----------------------------
# Figma: detect + fetch preview (public only)
# ----------------------------
FIGMA_URL_RE = re.compile(r"(https?://www\.figma\.com/(file|design)/[^\s]+)", re.IGNORECASE)

def extract_figma_url(text: str) -> Optional[str]:
    m = FIGMA_URL_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip()

def figma_oembed_url(figma_url: str) -> str:
    bust = int(time.time() * 1000)
    return "https://www.figma.com/oembed?url=" + urllib.parse.quote(figma_url, safe="") + f"&_cb={bust}"

def http_get_bytes(url: str, max_bytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError("Preview is too large.")
    return data

def fetch_figma_preview(figma_url: str) -> Tuple[bytes, str]:
    oembed = http_get_bytes(figma_oembed_url(figma_url), max_bytes=400_000).decode("utf-8", errors="replace")
    payload = json.loads(oembed)
    thumb = payload.get("thumbnail_url")
    title = payload.get("title") or "Figma"
    if not thumb:
        raise RuntimeError("Figma не дал thumbnail_url. Возможно, файл не публичный.")
    img_bytes = http_get_bytes(thumb, max_bytes=MAX_PREVIEW_BYTES)
    caption = f"{title}\n(превью из Figma)"
    return img_bytes, caption


# ----------------------------
# OpenAI: Vision review (structured)
# ----------------------------
REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "what_i_see": {"type": "string"},
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 10},

        "visual_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "visual_praise": {"type": "array", "items": {"type": "string"}},
        "visual_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "why_bad": {"type": "string"},
                    "how_to_fix": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["title", "why_bad", "how_to_fix", "severity"],
            },
        },

        "copy_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "copy_praise": {"type": "array", "items": {"type": "string"}},
        "copy_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "what_wrong": {"type": "string"},
                    "rewrite": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "role_guess": {"type": "string", "enum": ["header", "button", "body", "hint", "error", "label", "unknown"]},
                },
                "required": ["title", "what_wrong", "rewrite", "severity", "role_guess"],
            },
        },

        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "n": {"type": "integer", "minimum": 1, "maximum": 99},
                    "label": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "center": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1}, "y": {"type": "number", "minimum": 0, "maximum": 1}},
                        "required": ["x", "y"],
                    },
                    "bbox": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "x": {"type": "number", "minimum": 0, "maximum": 1},
                            "y": {"type": "number", "minimum": 0, "maximum": 1},
                            "w": {"type": "number", "minimum": 0, "maximum": 1},
                            "h": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["x", "y", "w", "h"],
                    },
                },
                "required": ["n", "label", "severity", "center", "bbox"],
            },
        },

        "font_family_guess": {"type": "string"},
        "palette_vibe_guess": {"type": "string"},
    },
    "required": [
        "what_i_see",
        "overall_score",
        "visual_score",
        "visual_praise",
        "visual_issues",
        "copy_score",
        "copy_praise",
        "copy_issues",
        "annotations",
        "font_family_guess",
        "palette_vibe_guess",
    ],
}

SYSTEM_STYLE = (
    "Ты — старший товарищ по дизайн-ревью. "
    "Если плохо — говори прямо и придирчиво, но без мата и без унижений. "
    "Если хорошо — хвали и называй конкретно, что именно хорошо. "
    "НЕ добавляй технические замеры: никаких px, чисел размеров, RGB/hex. "
    "Про шрифты и палитру — только качественные догадки (семейство/вайб). "
    "Не выдумывай элементы, которых не видно. "
    "Рекомендации — применимые."
)

TASK_PROMPT = (
    "Сделай ревью интерфейса по скриншоту.\n\n"
    "Верни JSON строго по схеме.\n\n"
    "Критично:\n"
    "- Аннотации ставь ТОЛЬКО на реальные элементы (текст/кнопка/поле/лейбл/сообщение). НЕ на пустые зоны.\n"
    "- Если не уверен в роли текста — role_guess='unknown'.\n"
    "- bbox и center должны быть согласованны: center внутри bbox.\n"
    "- Если не уверен в точных координатах — лучше меньше аннотаций, но релевантных.\n"
)

def _b64_png_from_bytes(img_bytes: bytes) -> str:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def call_llm_review(img_bytes: bytes) -> Dict[str, Any]:
    img_b64 = _b64_png_from_bytes(img_bytes)
    data_url = "data:image/png;base64," + img_b64

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_STYLE}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": TASK_PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ui_review",
                "schema": REVIEW_SCHEMA,
            }
        },
    )

    raw = (getattr(resp, "output_text", "") or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError("LLM вернул невалидный JSON. Пришли тот же скрин ещё раз (или чуть крупнее).")


# ----------------------------
# Annotation drawing: pins + glow
# ----------------------------
def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def sev_tag(sev: str) -> str:
    return {"high": "!", "medium": "~", "low": "."}.get(sev, ".")

def draw_annotations(img_bytes: bytes, ann: List[Dict[str, Any]]) -> bytes:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = im.size

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def expand_box(x, y, bw, bh, pad=0.02):
        px = int(pad * w)
        py = int(pad * h)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w, x + bw + px)
        y2 = min(h, y + bh + py)
        return x1, y1, x2, y2

    glow_fill = (255, 255, 255, 35)
    glow_outline = (255, 255, 255, 120)
    pin_fill = (0, 0, 0, 255)
    pin_outline = (255, 255, 255, 255)
    text_bg = (0, 0, 0, 255)
    text_fg = (255, 255, 255, 255)

    for a in (ann or [])[:40]:
        if not isinstance(a, dict):
            continue

        bbox = a.get("bbox") or {}
        center = a.get("center") or {}

        bx = int(clamp01(float(bbox.get("x", 0))) * w)
        by = int(clamp01(float(bbox.get("y", 0))) * h)
        bw = int(clamp01(float(bbox.get("w", 0))) * w)
        bh = int(clamp01(float(bbox.get("h", 0))) * h)

        cx = int(clamp01(float(center.get("x", 0))) * w)
        cy = int(clamp01(float(center.get("y", 0))) * h)

        if bw < 6 or bh < 6:
            bw = int(0.08 * w)
            bh = int(0.05 * h)
            bx = max(0, cx - bw // 2)
            by = max(0, cy - bh // 2)

        x1, y1, x2, y2 = expand_box(bx, by, bw, bh, pad=0.015)

        n = str(a.get("n", "?"))
        sev = str(a.get("severity", "low"))
        label = (str(a.get("label", "")).strip() or "Check")

        draw.rectangle([x1, y1, x2, y2], fill=glow_fill, outline=glow_outline, width=2)

        r = 8
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=pin_fill, outline=pin_outline, width=2)
        draw.line([cx - 14, cy, cx + 14, cy], fill=pin_outline, width=1)
        draw.line([cx, cy - 14, cx, cy + 14], fill=pin_outline, width=1)

        tag = f"{n}{sev_tag(sev)}"
        tx = min(w - 220, max(0, cx + 14))
        ty = min(h - 16, max(0, cy - 12))
        draw.rectangle([tx, ty, tx + 220, ty + 16], fill=text_bg)
        txt = f"{tag} {label[:60]}"
        draw.text((tx + 4, ty + 2), txt, fill=text_fg, font=font)

    out = Image.alpha_composite(im, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------
# Formatting
# ----------------------------
def bullet(lines: List[str], limit: int = 8) -> str:
    out = []
    for s in lines[:limit]:
        s = (s or "").strip()
        if s:
            out.append(f"- {s}")
    return "\n".join(out) if out else "- (нет)"

def build_verdict(review: Dict[str, Any]) -> str:
    overall = int(review.get("overall_score", 5))
    vscore = int(review.get("visual_score", 5))
    cscore = int(review.get("copy_score", 5))

    font_guess = (review.get("font_family_guess") or "").strip()
    palette_guess = (review.get("palette_vibe_guess") or "").strip()

    vp = review.get("visual_praise") or []
    cp = review.get("copy_praise") or []

    vis_issues = [i for i in (review.get("visual_issues") or []) if isinstance(i, dict)]
    copy_issues = [i for i in (review.get("copy_issues") or []) if isinstance(i, dict)]

    def sort_key(i: Dict[str, Any]) -> int:
        sev = str(i.get("severity", "low"))
        return {"high": 0, "medium": 1, "low": 2}.get(sev, 3)

    vis_issues.sort(key=sort_key)
    copy_issues.sort(key=sort_key)

    lines = []
    lines.append("Вердикт")
    lines.append(f"Общая оценка: {overall}/10")
    lines.append(f"Визуал: {vscore}/10")
    lines.append(f"Текст: {cscore}/10")
    lines.append("")

    if font_guess or palette_guess:
        lines.append("Быстрые догадки:")
        if font_guess:
            lines.append(f"- Шрифт: похоже на {font_guess}")
        if palette_guess:
            lines.append(f"- Палитра: {palette_guess}")
        lines.append("")

    if vp or cp:
        lines.append("Что хорошо (оставляем и усиливаем):")
        if vp:
            lines.append("Визуал:")
            lines.append(bullet([str(x) for x in vp], limit=5))
        if cp:
            lines.append("Текст:")
            lines.append(bullet([str(x) for x in cp], limit=5))
        lines.append("")

    lines.append("Что чинить в первую очередь:")

    if vis_issues:
        lines.append("Визуал:")
        for i in vis_issues[:4]:
            sev = str(i.get("severity", "low"))
            st = {"high": "[!]", "medium": "[~]", "low": "[.]"}[sev]
            lines.append(
                f"{st} {str(i.get('title','')).strip() or 'Проблема'}\n"
                f"  Что не так: {str(i.get('why_bad','')).strip() or '—'}\n"
                f"  Как улучшить: {str(i.get('how_to_fix','')).strip() or '—'}"
            )
    else:
        lines.append("Визуал:\n- (нет)")

    if copy_issues:
        lines.append("")
        lines.append("Текст:")
        for i in copy_issues[:4]:
            sev = str(i.get("severity", "low"))
            st = {"high": "[!]", "medium": "[~]", "low": "[.]"}[sev]
            role = str(i.get("role_guess", "unknown"))
            lines.append(
                f"{st} {str(i.get('title','')).strip() or 'Проблема'} (роль: {role})\n"
                f"  Что не так: {str(i.get('what_wrong','')).strip() or '—'}\n"
                f"  Как лучше: {str(i.get('rewrite','')).strip() or '—'}"
            )
    else:
        lines.append("\nТекст:\n- (нет)")

    return "\n".join(lines).strip()


# ----------------------------
# Concept generation
# ----------------------------
def generate_concept_image(review: Dict[str, Any]) -> Optional[bytes]:
    if not CONCEPT_ENABLED:
        return None

    what = (review.get("what_i_see") or "").strip()
    visual_issues = review.get("visual_issues") or []
    copy_issues = review.get("copy_issues") or []

    top_visual = "; ".join([i.get("title", "") for i in visual_issues[:5] if isinstance(i, dict)])
    top_copy = "; ".join([i.get("title", "") for i in copy_issues[:5] if isinstance(i, dict)])

    prompt = (
        "UI redesign concept of the same screen. "
        "Keep the same purpose and content, improve hierarchy, spacing, clarity, readability. "
        "Business UI, calm, no flashy decorations. "
        "Do not invent new features; refine layout and microcopy.\n"
        f"Observed: {what}\n"
        f"Fix visuals: {top_visual}\n"
        f"Fix copy: {top_copy}\n"
        "Output: a clean UI screenshot-like concept."
    )

    res = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        response_format="b64_json",
    )
    b64_out = getattr(res.data[0], "b64_json", None)
    if not b64_out:
        return None
    return base64.b64decode(b64_out)


# ----------------------------
# Core processing
# ----------------------------
async def process_image_and_reply(msg: Message, img_bytes: bytes, source_note: Optional[str] = None) -> None:
    await animate_progress(msg, title="Review in progress")

    review = call_llm_review(img_bytes)

    if source_note:
        await msg.answer(source_note)

    what = (review.get("what_i_see") or "").strip()
    await msg.answer(what if what else "(Не смог уверенно понять, что на экране)")

    await msg.answer(build_verdict(review))

    ann = review.get("annotations") or []
    try:
        annotated = draw_annotations(img_bytes, ann if isinstance(ann, list) else [])
        await msg.answer_photo(
            BufferedInputFile(annotated, filename="annotations.png"),
            caption="Аннотации (метки: номер + label)",
        )
    except Exception:
        await msg.answer("Аннотации не нарисовал: сломалось на этапе разметки картинки.")

    if CONCEPT_ENABLED:
        await animate_progress(msg, title="Concept build")
        try:
            concept = generate_concept_image(review)
            if concept:
                await msg.answer_photo(
                    BufferedInputFile(concept, filename="concept.png"),
                    caption="Концепт (черновик направления, без финальной полировки)",
                )
            else:
                await msg.answer("Концепт сейчас не сгенерировался.")
        except Exception:
            await msg.answer("Концепт сейчас не сгенерировался (сервис генерации недоступен).")


# ----------------------------
# Handlers
# ----------------------------
@router.message(F.text == "/start")
async def start_cmd(m: Message):
    await m.answer(WELCOME_TEXT, reply_markup=main_menu())

@router.callback_query(F.data == BTN_REVIEW)
async def cb_review(c: CallbackQuery):
    await c.message.answer("Кидай скриншот или ссылку на публичный Figma фрейм.", reply_markup=main_menu())
    await c.answer()

@router.callback_query(F.data == BTN_HOW)
async def cb_how(c: CallbackQuery):
    await c.message.answer(HOW_TEXT, reply_markup=main_menu())
    await c.answer()

@router.callback_query(F.data == BTN_PING)
async def cb_ping(c: CallbackQuery):
    await c.message.answer(PING_TEXT, reply_markup=main_menu())
    await c.answer()

@router.message(F.photo)
async def on_photo(m: Message):
    ph = m.photo[-1]
    file = await m.bot.get_file(ph.file_id)
    data = await m.bot.download_file(file.file_path)
    img_bytes = data.read()
    await process_image_and_reply(m, img_bytes)

@router.message(F.document)
async def on_document(m: Message):
    doc = m.document
    if not doc or not (doc.mime_type or "").startswith("image/"):
        await m.answer("Я беру на ревью только картинки (или ссылку на Figma фрейм).", reply_markup=main_menu())
        return
    file = await m.bot.get_file(doc.file_id)
    data = await m.bot.download_file(file.file_path)
    img_bytes = data.read()
    await process_image_and_reply(m, img_bytes)

@router.message(F.text)
async def on_text(m: Message):
    url = extract_figma_url(m.text or "")
    if not url:
        await m.answer(WELCOME_TEXT, reply_markup=main_menu())
        return

    try:
        preview_bytes, caption = fetch_figma_preview(url)
        await m.answer_photo(BufferedInputFile(preview_bytes, filename="figma_preview.png"), caption=caption)

        note = "Источник: Figma превью."
        await process_image_and_reply(m, preview_bytes, source_note=note)
    except Exception:
        await m.answer(
            "Не смог скачать превью из Figma.\nПроверь, что файл публичный, и попробуй ещё раз.",
            reply_markup=main_menu(),
        )


# ----------------------------
# Entrypoint
# ----------------------------
async def main():
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())