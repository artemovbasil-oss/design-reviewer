# bot.py
# Telegram UI/UX + Copy design-review buddy
# - accepts screenshots (images) and public Figma frame links (oEmbed preview)
# - outputs: (1) description (2) verdict+recs (3) annotated image (4) concept image
# - retro ASCII progress
# - no dotenv dependency

import asyncio
import base64
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
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

# Model for vision+text
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

# Concept image generation (set to "true" to enable)
CONCEPT_ENABLED = os.getenv("CONCEPT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y")

# Image generation model (per OpenAI docs; adjust if you use another)
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()

# Safety: max image bytes to download for figma preview
MAX_PREVIEW_BYTES = int(os.getenv("MAX_PREVIEW_BYTES", "8000000"))  # 8 MB

# Retro progress timing
PROGRESS_STEPS = int(os.getenv("PROGRESS_STEPS", "10"))
PROGRESS_DELAY = float(os.getenv("PROGRESS_DELAY", "0.20"))

# If empty -> fail fast
if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in environment variables (Railway Variables or local env).")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment variables (Railway Variables or local env).")

client = OpenAI(api_key=OPENAI_API_KEY)

router = Router()


# ----------------------------
# UI: monochrome style
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
# Helpers: robust message edit
# ----------------------------
async def safe_edit_or_resend(msg: Message, text: str, parse_mode: Optional[str] = None) -> Message:
    """
    Telegram sometimes refuses editing ("message can't be edited").
    This helper tries edit, otherwise sends a new message and returns it.
    """
    try:
        await msg.edit_text(text, parse_mode=parse_mode)
        return msg
    except Exception:
        # Fall back: send a new message
        try:
            return await msg.answer(text, parse_mode=parse_mode)
        except Exception:
            # As last resort, ignore
            return msg


def ascii_bar(i: int, total: int) -> str:
    # Retro, clean, no weird glyphs.
    filled = int((i / total) * 20)
    return "[" + "#" * filled + "-" * (20 - filled) + "]"


async def animate_progress(anchor: Message, title: str = "Смотрю внимательно") -> Message:
    """
    Shows a short retro ASCII animation by editing one message.
    Returns the last message used (might be re-sent if edit fails).
    """
    msg = await anchor.answer(f"{title}\n<code>{ascii_bar(0, PROGRESS_STEPS)}</code>", parse_mode=ParseMode.HTML)
    for i in range(1, PROGRESS_STEPS + 1):
        frame = f"{title}\n<code>{ascii_bar(i, PROGRESS_STEPS)}</code>"
        msg = await safe_edit_or_resend(msg, frame, parse_mode=ParseMode.HTML)
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
    # Add cache-buster so different node-id links don't get stuck.
    # Still depends on Figma returning node-specific preview; for some files it may not.
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
    """
    Returns (image_bytes, caption).
    Uses oEmbed thumbnail_url. Works for public files. If file is private, will fail.
    """
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
# OpenAI: Vision review with structured output
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
                    # role helps reduce "header vs button" confusion:
                    "role_guess": {"type": "string", "enum": ["header", "button", "body", "hint", "error", "label", "unknown"]},
                },
                "required": ["title", "what_wrong", "rewrite", "severity", "role_guess"],
            },
        },

        # For annotations: normalized bboxes 0..1, only where there is real text / UI element
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "n": {"type": "integer", "minimum": 1, "maximum": 99},
                    "label": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
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
                "required": ["n", "label", "severity", "bbox"],
            },
        },

        # Soft guesses, no numbers/colors:
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
    "НЕ добавляй технические замеры: никаких px, чисел размеров, RGB/hex, медиан. "
    "Про шрифты и палитру — только качественные догадки (семейство/вайб). "
    "Не выдумывай элементы, которых не видно. "
    "Не размазывайся — рекомендации должны быть применимыми."
)

TASK_PROMPT = (
    "Сделай ревью интерфейса по скриншоту.\n\n"
    "Нужно вернуть JSON строго по схеме.\n\n"
    "Правила:\n"
    "- Аннотации (bbox) ставь ТОЛЬКО на области, где реально есть текст/элемент (не пустые зоны).\n"
    "- Если не уверен в роли текста, ставь role_guess='unknown'.\n"
    "- Тон: требовательный, прямой, но без токсичности.\n"
    "- Что исправить: конкретика (переписать, укоротить, уточнить, сделать последовательнее, снять двусмысленность).\n"
)


def _b64_png_from_bytes(img_bytes: bytes) -> str:
    # ensure PNG for stable decode
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
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_STYLE}],
            },
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

    # openai Responses returns text in output_text convenience; keep robust:
    raw = getattr(resp, "output_text", None)
    if not raw:
        # fallback: try to stitch output parts
        raw = ""
        try:
            for o in resp.output:
                for c in getattr(o, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        raw += getattr(c, "text", "") or ""
        except Exception:
            pass

    raw = (raw or "").strip()

    try:
        return json.loads(raw)
    except Exception:
        # If model returns something weird (rare), try to extract first JSON block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError("LLM вернул невалидный JSON. Пришли тот же скрин ещё раз (или чуть крупнее).")


# ----------------------------
# Annotations: draw markers
# ----------------------------
def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def severity_char(sev: str) -> str:
    # monochrome markers
    return {"high": "!", "medium": "~", "low": "."}.get(sev, ".")

def draw_annotations(img_bytes: bytes, ann: List[Dict[str, Any]]) -> bytes:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)

    # Try to load a default font; fallback to PIL built-in.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw numbered boxes
    for a in ann[:60]:
        bbox = a.get("bbox") or {}
        x = int(clamp01(float(bbox.get("x", 0))) * w)
        y = int(clamp01(float(bbox.get("y", 0))) * h)
        bw = int(clamp01(float(bbox.get("w", 0))) * w)
        bh = int(clamp01(float(bbox.get("h", 0))) * h)

        # Avoid degenerate boxes
        if bw < 6 or bh < 6:
            continue

        x2, y2 = x + bw, y + bh
        n = str(a.get("n", "?"))
        sev = str(a.get("severity", "low"))

        # Simple high-contrast black/white
        draw.rectangle([x, y, x2, y2], outline="white", width=3)
        draw.rectangle([x+1, y+1, x2-1, y2-1], outline="black", width=1)

        tag = f"{n}{severity_char(sev)}"
        # label background
        tx, ty = x + 4, max(0, y - 14)
        draw.rectangle([tx - 2, ty - 2, tx + 40, ty + 14], fill="black")
        draw.text((tx, ty), tag, fill="white", font=font)

    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


# ----------------------------
# Concept image generation
# ----------------------------
def generate_concept_image(review: Dict[str, Any], base_img_bytes: bytes) -> Optional[bytes]:
    if not CONCEPT_ENABLED:
        return None

    # Keep it practical: produce a "cleaner" concept UI.
    # We avoid exact brand colors/sizes and focus on structure.
    what = (review.get("what_i_see") or "").strip()
    visual_issues = review.get("visual_issues") or []
    copy_issues = review.get("copy_issues") or []

    # Build a tight prompt (no explicit color codes/sizes)
    top_visual = "; ".join([i.get("title","") for i in visual_issues[:5] if isinstance(i, dict)])
    top_copy = "; ".join([i.get("title","") for i in copy_issues[:5] if isinstance(i, dict)])

    prompt = (
        "Create a redesigned concept of the same UI screen from the provided screenshot. "
        "Keep the same purpose and content, but improve clarity, hierarchy, spacing, and readability. "
        "No flashy decorations. Business product UI. "
        "Do not invent new features; refine layout and microcopy. "
        f"Observed screen: {what}\n"
        f"Main visual issues to address: {top_visual}\n"
        f"Main copy issues to address: {top_copy}\n"
        "Output: a clean, modern UI concept screenshot-style image."
    )

    # Use Images API (OpenAI docs). GPT image models return base64.
    # https://platform.openai.com/docs/api-reference/images
    img_b64 = _b64_png_from_bytes(base_img_bytes)
    data_url = "data:image/png;base64," + img_b64

    res = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        # Provide reference image if supported by your plan/model via prompt only (safe default).
        # If your setup supports edits, you can switch to image edits flow later.
    )

    # SDK returns base64 for GPT image models
    b64_out = None
    try:
        b64_out = res.data[0].b64_json
    except Exception:
        pass

    if not b64_out:
        return None
    return base64.b64decode(b64_out)


# ----------------------------
# Formatting: output messages
# ----------------------------
def fmt_score_line(name: str, score: int) -> str:
    # monochrome, no colored emoji
    return f"{name}: {score}/10"

def bullet(lines: List[str], limit: int = 8) -> str:
    out = []
    for s in lines[:limit]:
        s = (s or "").strip()
        if not s:
            continue
        out.append(f"- {s}")
    return "\n".join(out) if out else "- (нет)"

def fmt_issue_block(title: str, why: str, how: str, sev: str) -> str:
    sev_tag = {"high": "[!]", "medium": "[~]", "low": "[.]"}[sev]
    return (
        f"{sev_tag} {title}\n"
        f"  Что не так: {why}\n"
        f"  Как улучшить: {how}"
    )

def build_verdict(review: Dict[str, Any]) -> str:
    overall = int(review.get("overall_score", 5))
    vscore = int(review.get("visual_score", 5))
    cscore = int(review.get("copy_score", 5))

    font_guess = (review.get("font_family_guess") or "").strip()
    palette_guess = (review.get("palette_vibe_guess") or "").strip()

    vp = review.get("visual_praise") or []
    cp = review.get("copy_praise") or []

    vis_issues = review.get("visual_issues") or []
    copy_issues = review.get("copy_issues") or []

    # Build strict but fair tone
    lines = []
    lines.append("Вердикт")
    lines.append(fmt_score_line("Общая оценка", overall))
    lines.append(fmt_score_line("Визуал", vscore))
    lines.append(fmt_score_line("Текст", cscore))
    lines.append("")

    if font_guess or palette_guess:
        lines.append("Быстрые догадки (без занудства):")
        if font_guess:
            lines.append(f"- Шрифт: похоже на {font_guess}")
        if palette_guess:
            lines.append(f"- Палитра: {palette_guess}")
        lines.append("")

    # Praise first (keeps it buddy-like)
    if vp or cp:
        lines.append("Что хорошо (и это стоит сохранить):")
        if vp:
            lines.append("Визуал:")
            lines.append(bullet([str(x) for x in vp], limit=5))
        if cp:
            lines.append("Текст:")
            lines.append(bullet([str(x) for x in cp], limit=5))
        lines.append("")

    lines.append("Что чинить в первую очередь:")
    # Prioritize high severity issues
    def sort_key(i: Dict[str, Any]) -> int:
        sev = str(i.get("severity", "low"))
        return {"high": 0, "medium": 1, "low": 2}.get(sev, 3)

    # Visual issues
    vis_sorted = [i for i in vis_issues if isinstance(i, dict)]
    vis_sorted.sort(key=sort_key)
    if vis_sorted:
        lines.append("Визуал:")
        for i in vis_sorted[:4]:
            lines.append(fmt_issue_block(
                str(i.get("title","")).strip() or "Проблема",
                str(i.get("why_bad","")).strip() or "—",
                str(i.get("how_to_fix","")).strip() or "—",
                str(i.get("severity","low")),
            ))
    else:
        lines.append("Визуал:\n- (нет)")

    # Copy issues
    copy_sorted = [i for i in copy_issues if isinstance(i, dict)]
    copy_sorted.sort(key=sort_key)
    if copy_sorted:
        lines.append("")
        lines.append("Текст:")
        for i in copy_sorted[:4]:
            title = str(i.get("title","")).strip() or "Проблема"
            wrong = str(i.get("what_wrong","")).strip() or "—"
            rewrite = str(i.get("rewrite","")).strip() or "—"
            sev = str(i.get("severity","low"))
            sev_tag = {"high": "[!]", "medium": "[~]", "low": "[.]"}[sev]
            role = str(i.get("role_guess","unknown"))
            lines.append(
                f"{sev_tag} {title} (роль: {role})\n"
                f"  Что не так: {wrong}\n"
                f"  Как лучше: {rewrite}"
            )
    else:
        lines.append("\nТекст:\n- (нет)")

    return "\n".join(lines).strip()


# ----------------------------
# Core processing
# ----------------------------
async def process_image_and_reply(msg: Message, img_bytes: bytes, source_note: Optional[str] = None) -> None:
    # Progress 1
    await animate_progress(msg, title="Смотрю внимательно")

    review = call_llm_review(img_bytes)

    # 1) what i see
    what = (review.get("what_i_see") or "").strip()
    if source_note:
        await msg.answer(source_note)
    await msg.answer(what if what else "(не смог разобрать, что на экране)")

    # 2) verdict
    verdict = build_verdict(review)
    await msg.answer(verdict)

    # 3) annotated screenshot
    ann = review.get("annotations") or []
    try:
        annotated = draw_annotations(img_bytes, ann if isinstance(ann, list) else [])
        await msg.answer_photo(
            BufferedInputFile(annotated, filename="annotations.png"),
            caption="Аннотации на скрине (метки = куда смотреть)",
        )
    except Exception:
        await msg.answer("Аннотации не нарисовал: что-то пошло не так при разметке картинки.")

    # Progress 2 (before concept)
    if CONCEPT_ENABLED:
        await animate_progress(msg, title="Собираю концепт")

        concept = None
        try:
            concept = generate_concept_image(review, img_bytes)
        except Exception:
            concept = None

        if concept:
            await msg.answer_photo(
                BufferedInputFile(concept, filename="concept.png"),
                caption="Концепт: как могло бы быть (черновик направления)",
            )
        else:
            await msg.answer("Концепт не сгенерировался (модель/лимиты/формат).")


# ----------------------------
# Handlers
# ----------------------------
@router.message(F.text == "/start")
async def start_cmd(m: Message):
    await m.answer(WELCOME_TEXT, reply_markup=main_menu())

@router.callback_query(F.data == BTN_REVIEW)
async def cb_review(c: CallbackQuery):
    await c.message.answer("Ок. Кидай скриншот или ссылку на публичный Figma фрейм.", reply_markup=main_menu())
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
    # Take best quality photo
    ph = m.photo[-1]
    file = await m.bot.get_file(ph.file_id)
    data = await m.bot.download_file(file.file_path)
    img_bytes = data.read()
    await process_image_and_reply(m, img_bytes)

@router.message(F.document)
async def on_document(m: Message):
    # Allow image as document
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
        await m.answer(
            "Я принимаю на ревью:\n- скриншоты (картинки)\n- ссылки на Figma фреймы (если файл публичный)\n\n"
            "Жми «Закинуть на ревью» или просто отправь скрин/ссылку.",
            reply_markup=main_menu(),
        )
        return

    # Try to fetch preview
    try:
        preview_bytes, caption = fetch_figma_preview(url)
        await m.answer_photo(
            BufferedInputFile(preview_bytes, filename="figma_preview.png"),
            caption=caption,
        )
        note = "Источник: Figma превью (если у Figma одно превью на весь файл — разные node-id могут выглядеть одинаково)."
        await process_image_and_reply(m, preview_bytes, source_note=note)
    except Exception as e:
        await m.answer(
            f"Не смог скачать превью из Figma.\nПричина: {str(e)}\n\n"
            "Проверь, что файл публичный, и попробуй ещё раз.",
            reply_markup=main_menu(),
        )


# ----------------------------
# Entrypoint
# ----------------------------
async def main():
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())