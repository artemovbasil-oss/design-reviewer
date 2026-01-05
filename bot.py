#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Design Reviewer Telegram Bot (aiogram 3.7+)
- Accepts screenshots (photos/images) OR public Figma frame links
- Runs an LLM review and sends:
  1) What I see (description)
  2) Verdict + recommendations (UX + Text together) + score/10
  3) Annotated screenshot (boxes)
  4) ASCII concept as a BEAUTIFUL retro image (same size as screenshot)

UI:
- Main menu appears ONLY after /start and at the END of each review
- During processing: compact retro ASCII progress animation + Cancel button
- Bottom buttons: Submit for review / How it works? / (Channel + Language toggle in one row)
- English default, RU optional toggle

No dotenv, no requests/httpx. Uses aiohttp (comes with aiogram).
"""

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

# OpenAI SDK (openai==2.x)
from openai import OpenAI


# ---------------------------
# ENV / CONFIG
# ---------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN environment variable")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

# Keep it safe: send as plain text or <pre>. We'll use HTML globally and escape ourselves.
BOT_DEFAULT = DefaultBotProperties(parse_mode=ParseMode.HTML)

# Channel
CHANNEL_HANDLE = "@prodooktovy"
CHANNEL_URL = f"https://t.me/{CHANNEL_HANDLE.lstrip('@')}"

# Supported languages
LANG_EN = "en"
LANG_RU = "ru"

# In-memory user state (simple + fine for Railway)
USER_LANG: Dict[int, str] = {}
RUNNING_TASKS: Dict[int, asyncio.Task] = {}
CANCEL_FLAGS: Dict[int, bool] = {}

# Progress animation tuning (compact + fast)
PROGRESS_STEPS = 10
PROGRESS_DELAY = 0.12  # seconds
PROGRESS_WIDTH = 18    # compact

# ASCII concept image styling
PADDING = 36
LINE_SPACING = 6
BG = "#000000"
FG = "#00FF88"
FG_DIM = "#88AA99"
BORDER = "#2B3B33"
TITLE_FG = "#B7FFE3"
MAX_CONCEPT_COLS = 54  # safe-ish for phone look; image will be same size anyway


FONT_CANDIDATES = [
    "fonts/PxPlus_IBM_VGA9.ttf",
    "fonts/TerminusTTF.ttf",
    "fonts/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Railway fallback
]


# ---------------------------
# I18N TEXTS
# ---------------------------

TEXTS = {
    LANG_EN: {
        "hello_title": "Design Reviewer is online.",
        "hello_body": (
            "Send me a UI screenshot <b>or</b> a public Figma frame link.\n"
            "I’ll be picky (in a helpful way) and show you exactly what to fix."
        ),
        "menu_submit": "Submit for review",
        "menu_how": "How does it work?",
        "menu_channel": "Product design channel",
        "menu_lang": "EN/RU",
        "how": (
            "1) Tap <b>Submit for review</b> (or just send a screenshot / Figma link)\n"
            "2) Wait for the review\n"
            "3) Get: what I see → verdict → annotated screenshot → ASCII concept image"
        ),
        "send_prompt": "Send a screenshot (image) or a public Figma frame link.",
        "bad_link": "That doesn’t look like a Figma link. Send a screenshot or a Figma URL.",
        "figma_not_public": "I couldn’t get a public preview from that Figma link. Make the file public (or send a screenshot).",
        "cancel": "Cancel",
        "cancelled": "Cancelled. Send a new screenshot/link when ready.",
        "llm_error": "LLM error. Try again (or send a clearer / larger screenshot).",
        "review_done": "Review done.",
        "score": "Score",
        "what_i_see": "What I see",
        "verdict": "Verdict & recommendations",
        "annotated": "Annotated screenshot",
        "concept": "ASCII concept",
    },
    LANG_RU: {
        "hello_title": "Design Reviewer на связи.",
        "hello_body": (
            "Кидай скрин интерфейса <b>или</b> ссылку на публичный Figma-фрейм.\n"
            "Я докопаюсь по делу и покажу, что именно улучшать."
        ),
        "menu_submit": "Закинуть на ревью",
        "menu_how": "Как это работает?",
        "menu_channel": "Канал о продуктовом дизайне",
        "menu_lang": "EN/RU",
        "how": (
            "1) Жми <b>Закинуть на ревью</b> (или просто отправь скрин/ссылку)\n"
            "2) Подожди ревью\n"
            "3) Получишь: что вижу → вердикт → аннотации → ASCII-концепт"
        ),
        "send_prompt": "Отправь скриншот (картинку) или ссылку на публичный Figma-фрейм.",
        "bad_link": "Это не похоже на ссылку Figma. Отправь скрин или URL на Figma.",
        "figma_not_public": "Не смог достать публичное превью Figma. Сделай файл публичным (или пришли скрин).",
        "cancel": "Отмена",
        "cancelled": "Ок, отменил. Пришли новый скрин/ссылку, когда будешь готов.",
        "llm_error": "Ошибка LLM. Попробуй ещё раз (или пришли скрин крупнее/четче).",
        "review_done": "Ревью готово.",
        "score": "Оценка",
        "what_i_see": "Что вижу",
        "verdict": "Вердикт и рекомендации",
        "annotated": "Аннотированный скрин",
        "concept": "ASCII-концепт",
    },
}


def t(uid: int, key: str) -> str:
    lang = USER_LANG.get(uid, LANG_EN)
    return TEXTS.get(lang, TEXTS[LANG_EN]).get(key, key)


# ---------------------------
# HTML SAFE
# ---------------------------

def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


# ---------------------------
# KEYBOARDS
# ---------------------------

def main_menu_kb(uid: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text=t(uid, "menu_submit"), callback_data="menu_submit"))
    b.row(InlineKeyboardButton(text=t(uid, "menu_how"), callback_data="menu_how"))
    # last row: channel + language (same row if possible)
    b.row(
        InlineKeyboardButton(text=t(uid, "menu_channel"), url=CHANNEL_URL),
        InlineKeyboardButton(text=t(uid, "menu_lang"), callback_data="toggle_lang"),
    )
    return b.as_markup()


def cancel_kb(uid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=t(uid, "cancel"), callback_data="cancel")]]
    )


# ---------------------------
# FIGMA FETCH (PUBLIC PREVIEW)
# ---------------------------

FIGMA_RE = re.compile(r"https?://(www\.)?figma\.com/.*", re.IGNORECASE)

def normalize_figma_url(url: str) -> str:
    # Keep node-id if present; remove trailing spaces
    return url.strip()

async def figma_oembed_thumbnail(session: aiohttp.ClientSession, figma_url: str) -> Optional[str]:
    """
    Uses Figma oEmbed to get a thumbnail URL (works only for public-accessible links).
    """
    # Official-ish oEmbed endpoint used by many integrations:
    # https://www.figma.com/api/oembed?url=<figma_url>
    endpoint = "https://www.figma.com/api/oembed"
    full = f"{endpoint}?{urlencode({'url': figma_url})}"
    try:
        async with session.get(full, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200:
                return None
            data = await r.json()
            # thumbnail_url is usually present
            return data.get("thumbnail_url") or data.get("thumbnail_url_with_size")
    except Exception:
        return None

async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                return None
            return await r.read()
    except Exception:
        return None


# ---------------------------
# RETRO PROGRESS ANIMATION (COMPACT)
# ---------------------------

def retro_bar(step: int, total: int, width: int = PROGRESS_WIDTH) -> str:
    """
    Compact retro bar. Example:
    [####------] 40%
    """
    filled = int(width * step / total)
    empty = width - filled
    pct = int(100 * step / total)
    return f"[{'#' * filled}{'-' * empty}] {pct:3d}%"

def spinner_frame(i: int) -> str:
    frames = ["|", "/", "-", "\\"]
    return frames[i % len(frames)]

async def animate_progress(anchor: Message, uid: int, title: str) -> Message:
    """
    Sends ONE message and edits it quickly with a compact retro bar.
    If edit fails ('message can't be edited'), we fallback by sending a new message.
    Returns the message object used for animation.
    """
    msg = await anchor.answer(
        f"{esc(title)}\n<pre>{esc(retro_bar(0, PROGRESS_STEPS))}</pre>",
        reply_markup=cancel_kb(uid),
    )
    for i in range(1, PROGRESS_STEPS + 1):
        if CANCEL_FLAGS.get(uid):
            break
        text = f"{esc(title)} {esc(spinner_frame(i))}\n<pre>{esc(retro_bar(i, PROGRESS_STEPS))}</pre>"
        try:
            await msg.edit_text(text, reply_markup=cancel_kb(uid))
        except Exception:
            # Can't edit -> send a new one and continue editing that
            try:
                msg = await anchor.answer(text, reply_markup=cancel_kb(uid))
            except Exception:
                pass
        await asyncio.sleep(PROGRESS_DELAY)
    return msg


# ---------------------------
# LLM CALL
# ---------------------------

@dataclass
class Box:
    x: float  # 0..1
    y: float
    w: float
    h: float
    label: str

@dataclass
class ReviewResult:
    description: str
    verdict: str
    score: int
    boxes: List[Box]
    concept_ascii: str


SYSTEM_PROMPT = """You are a senior product designer doing a ruthless-but-helpful design review.
Tone:
- direct, picky, no profanity
- praise specific things that are good
- criticize what is weak and propose concrete fixes
Do NOT output any markdown. Output JSON only.

You will receive a UI screenshot.
Return a JSON object with fields:
- description: string (what you see)
- score: integer 1..10
- verdict: string (combined UX + copy recommendations in one message; structured with short paragraphs)
- boxes: array of {x,y,w,h,label} where x,y,w,h are RELATIVE (0..1) and label is short.
  Only add boxes when you are confident. Avoid random empty areas.
- concept_ascii: string. A compact retro ASCII concept illustration (max ~40 lines).
  Keep lines short (<= 54 chars). No emojis. No weird quotes/brackets.
"""

def looks_like_dict_string(s: str) -> bool:
    s = s.strip()
    return s.startswith("{") and s.endswith("}")

def coerce_json(text: str) -> Optional[dict]:
    """
    Extract JSON from model output safely.
    """
    if not text:
        return None
    t_ = text.strip()

    # Sometimes model wraps JSON with extra text — attempt to locate outermost {...}
    if not looks_like_dict_string(t_):
        m = re.search(r"\{.*\}", t_, re.DOTALL)
        if m:
            t_ = m.group(0).strip()

    try:
        return json.loads(t_)
    except Exception:
        return None

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def parse_review(data: dict) -> Optional[ReviewResult]:
    try:
        desc = str(data.get("description", "")).strip()
        verdict = str(data.get("verdict", "")).strip()
        score = int(data.get("score", 0))
        score = max(1, min(10, score))
        concept = str(data.get("concept_ascii", "")).strip()

        boxes_raw = data.get("boxes", []) or []
        boxes: List[Box] = []
        for b in boxes_raw:
            x = float(b.get("x", 0))
            y = float(b.get("y", 0))
            w = float(b.get("w", 0))
            h = float(b.get("h", 0))
            label = str(b.get("label", "")).strip()[:60]
            if w <= 0 or h <= 0:
                continue
            boxes.append(Box(clamp01(x), clamp01(y), clamp01(w), clamp01(h), label))

        if not desc or not verdict:
            return None
        return ReviewResult(description=desc, verdict=verdict, score=score, boxes=boxes, concept_ascii=concept)
    except Exception:
        return None

def sanitize_concept_lines(s: str, max_cols: int = MAX_CONCEPT_COLS) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("\t", "    ")

    lines = []
    for raw in s.splitlines():
        # allow box drawing chars + standard ascii
        line = "".join(
            ch for ch in raw
            if (32 <= ord(ch) <= 126) or ch in "─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬"
        )
        # hard wrap to max_cols
        while len(line) > max_cols:
            lines.append(line[:max_cols])
            line = line[max_cols:]
        lines.append(line)

    # trim
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)

async def llm_review_image(img_bytes: bytes) -> Optional[ReviewResult]:
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{img_b64}"

    # 2 tries: strict JSON only on retry
    for attempt in range(2):
        extra = ""
        if attempt == 1:
            extra = "\nIMPORTANT: Output ONLY valid JSON. No trailing text. No markdown."

        try:
            resp = client.responses.create(
                model=LLM_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT + extra}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Review this screenshot."},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    },
                ],
            )
            # openai Responses API returns text in output[0].content[*].text
            out_text = ""
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text", "summary_text"):
                        out_text += getattr(c, "text", "") or ""
            out_text = out_text.strip()

            data = coerce_json(out_text)
            if not data:
                continue
            rr = parse_review(data)
            if rr:
                rr.concept_ascii = sanitize_concept_lines(rr.concept_ascii, MAX_CONCEPT_COLS)
                return rr
        except Exception:
            continue

    return None


# ---------------------------
# DRAW ANNOTATIONS
# ---------------------------

def _load_mono_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def _text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Pillow-safe width/height
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return (x1 - x0, y1 - y0)
    except Exception:
        try:
            x0, y0, x1, y1 = font.getbbox(text)
            return (x1 - x0, y1 - y0)
        except Exception:
            return (len(text) * 8, 16)

def draw_annotations(img_bytes: bytes, boxes: List[Box]) -> Optional[bytes]:
    if not boxes:
        return None

    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    font = _load_mono_font(18)

    for i, b in enumerate(boxes[:20], start=1):
        x0 = int(b.x * W)
        y0 = int(b.y * H)
        x1 = int((b.x + b.w) * W)
        y1 = int((b.y + b.h) * H)

        # clamp
        x0 = max(0, min(W - 1, x0))
        y0 = max(0, min(H - 1, y0))
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        if x1 <= x0 or y1 <= y0:
            continue

        # box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=4)

        label = f"{i}. {b.label}" if b.label else f"{i}"
        tw, th = _text_wh(draw, label, font)
        pad = 6
        # label background
        bx0 = x0
        by0 = max(0, y0 - (th + pad * 2))
        bx1 = min(W, x0 + tw + pad * 2)
        by1 = by0 + th + pad * 2
        draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
        draw.text((bx0 + pad, by0 + pad), label, font=font, fill=(255, 255, 255))

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ---------------------------
# ASCII CONCEPT AS IMAGE
# ---------------------------

def render_ascii_concept_image(original_img_bytes: bytes, ascii_text: str, title: str) -> Optional[bytes]:
    """
    Creates a share-worthy retro image:
    - same size as original screenshot
    - black background
    - subtle border
    - retro mono font
    - title + ASCII concept rendered neatly
    """
    if not ascii_text.strip():
        return None

    base_img = Image.open(BytesIO(original_img_bytes)).convert("RGB")
    W, H = base_img.size

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Border frame
    draw.rectangle((10, 10, W - 10, H - 10), outline=BORDER, width=2)
    draw.rectangle((14, 14, W - 14, H - 14), outline="#16211C", width=1)

    # Auto-pick font size so ~MAX_CONCEPT_COLS fits
    size = 28
    font = _load_mono_font(size)
    # reduce until it fits horizontally
    while size >= 12:
        font = _load_mono_font(size)
        tw, _ = _text_wh(draw, "M" * MAX_CONCEPT_COLS, font)
        if tw <= (W - 2 * PADDING):
            break
        size -= 2

    title_font = _load_mono_font(max(12, int(size * 0.9)))

    # Title
    title_text = title.upper()
    draw.text((PADDING, PADDING - 6), title_text, font=title_font, fill=TITLE_FG)
    # Separator line
    y_sep = PADDING + 24
    draw.line((PADDING, y_sep, W - PADDING, y_sep), fill=BORDER, width=2)

    # Render concept lines
    lines = sanitize_concept_lines(ascii_text, MAX_CONCEPT_COLS).splitlines()
    y = y_sep + 18
    for line in lines[:80]:
        if y > H - PADDING - 10:
            break
        draw.text((PADDING, y), line, font=font, fill=FG)
        y += (size + LINE_SPACING)

    # Footer
    footer = "design-reviewer • retro concept"
    fw, fh = _text_wh(draw, footer, title_font)
    draw.text((W - PADDING - fw, H - PADDING - fh), footer, font=title_font, fill=FG_DIM)

    out = BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


async def send_png(m: Message, png_bytes: bytes, caption: str) -> None:
    bio = BytesIO(png_bytes)
    bio.seek(0)
    f = BufferedInputFile(bio.read(), filename="concept.png")
    await m.answer_photo(photo=f, caption=esc(caption))


# ---------------------------
# REVIEW PIPELINE
# ---------------------------

async def process_review_from_image(m: Message, img_bytes: bytes) -> None:
    uid = m.from_user.id

    # 1) compact progress animation
    await animate_progress(m, uid, title="Review in progress")

    if CANCEL_FLAGS.get(uid):
        await m.answer(esc(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
        return

    rr = await llm_review_image(img_bytes)
    if not rr:
        await m.answer(esc(t(uid, "llm_error")), reply_markup=main_menu_kb(uid))
        return

    if CANCEL_FLAGS.get(uid):
        await m.answer(esc(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
        return

    # 2) Message: description
    await m.answer(
        f"<b>{esc(t(uid, 'what_i_see'))}</b>\n{esc(rr.description)}",
        reply_markup=None,
    )

    # 3) Message: verdict + score
    await m.answer(
        f"<b>{esc(t(uid, 'verdict'))}</b>\n"
        f"<b>{esc(t(uid, 'score'))}:</b> {rr.score}/10\n\n"
        f"{esc(rr.verdict)}",
        reply_markup=None,
    )

    # 4) Annotated screenshot
    annotated_bytes = draw_annotations(img_bytes, rr.boxes) if rr.boxes else None
    if annotated_bytes:
        try:
            f = BufferedInputFile(annotated_bytes, filename="annotated.png")
            await m.answer_photo(photo=f, caption=esc(t(uid, "annotated")), reply_markup=None)
        except Exception:
            # If anything fails, just skip quietly
            pass

    # 5) Concept progress (short)
    await animate_progress(m, uid, title="Generating concept")

    if CANCEL_FLAGS.get(uid):
        await m.answer(esc(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
        return

    concept_img = render_ascii_concept_image(img_bytes, rr.concept_ascii, title=t(uid, "concept"))
    if concept_img:
        try:
            await send_png(m, concept_img, caption=t(uid, "concept"))
        except Exception:
            pass

    # End: show menu (ONLY at end of review)
    await m.answer(esc(t(uid, "review_done")), reply_markup=main_menu_kb(uid))


async def process_review_from_figma_url(m: Message, url: str) -> None:
    uid = m.from_user.id
    async with aiohttp.ClientSession() as session:
        thumb = await figma_oembed_thumbnail(session, url)
        if not thumb:
            await m.answer(esc(t(uid, "figma_not_public")), reply_markup=main_menu_kb(uid))
            return
        img_bytes = await fetch_bytes(session, thumb)
        if not img_bytes:
            await m.answer(esc(t(uid, "figma_not_public")), reply_markup=main_menu_kb(uid))
            return

    # Show preview image first (nice UX)
    try:
        f = BufferedInputFile(img_bytes, filename="figma_preview.png")
        await m.answer_photo(photo=f, caption=esc("Figma preview"), reply_markup=None)
    except Exception:
        pass

    await process_review_from_image(m, img_bytes)


# ---------------------------
# TASK MANAGEMENT (CANCEL)
# ---------------------------

def start_task(uid: int, coro) -> None:
    # cancel previous if any
    prev = RUNNING_TASKS.get(uid)
    if prev and not prev.done():
        prev.cancel()
    CANCEL_FLAGS[uid] = False
    RUNNING_TASKS[uid] = asyncio.create_task(coro)

def cancel_task(uid: int) -> None:
    CANCEL_FLAGS[uid] = True
    task = RUNNING_TASKS.get(uid)
    if task and not task.done():
        task.cancel()


# ---------------------------
# ROUTERS / HANDLERS
# ---------------------------

dp = Dispatcher()

@dp.message(Command("start"))
async def on_start(m: Message):
    uid = m.from_user.id
    USER_LANG.setdefault(uid, LANG_EN)
    await m.answer(
        f"<b>{esc(t(uid, 'hello_title'))}</b>\n{esc(t(uid, 'hello_body'))}",
        reply_markup=main_menu_kb(uid),
    )

@dp.callback_query(F.data == "menu_how")
async def on_how(cb: CallbackQuery):
    uid = cb.from_user.id
    await cb.answer()
    await cb.message.answer(esc(t(uid, "how")), reply_markup=None)

@dp.callback_query(F.data == "menu_submit")
async def on_submit(cb: CallbackQuery):
    uid = cb.from_user.id
    await cb.answer()
    await cb.message.answer(esc(t(uid, "send_prompt")), reply_markup=None)

@dp.callback_query(F.data == "toggle_lang")
async def on_toggle_lang(cb: CallbackQuery):
    uid = cb.from_user.id
    cur = USER_LANG.get(uid, LANG_EN)
    USER_LANG[uid] = LANG_RU if cur == LANG_EN else LANG_EN
    await cb.answer()
    # refresh menu text
    await cb.message.answer(
        f"<b>{esc(t(uid, 'hello_title'))}</b>\n{esc(t(uid, 'hello_body'))}",
        reply_markup=main_menu_kb(uid),
    )

@dp.callback_query(F.data == "cancel")
async def on_cancel(cb: CallbackQuery):
    uid = cb.from_user.id
    cancel_task(uid)
    await cb.answer()
    await cb.message.answer(esc(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))

@dp.message(F.photo)
async def on_photo(m: Message):
    uid = m.from_user.id
    # Download highest-res photo
    file = await m.bot.get_file(m.photo[-1].file_id)
    buf = BytesIO()
    await m.bot.download_file(file.file_path, destination=buf)
    img_bytes = buf.getvalue()

    start_task(uid, process_review_from_image(m, img_bytes))

@dp.message(F.document)
async def on_document(m: Message):
    uid = m.from_user.id
    doc = m.document
    if not doc or not (doc.mime_type or "").startswith("image/"):
        return
    file = await m.bot.get_file(doc.file_id)
    buf = BytesIO()
    await m.bot.download_file(file.file_path, destination=buf)
    img_bytes = buf.getvalue()

    start_task(uid, process_review_from_image(m, img_bytes))

@dp.message(F.text)
async def on_text(m: Message):
    uid = m.from_user.id
    text = (m.text or "").strip()

    # Figma link?
    if FIGMA_RE.match(text):
        url = normalize_figma_url(text)
        start_task(uid, process_review_from_figma_url(m, url))
        return

    # Otherwise, gentle reminder (no menu spam)
    # Only show menu if they already did /start? We'll just keep it minimal.
    await m.answer(esc(t(uid, "send_prompt")), reply_markup=None)


# ---------------------------
# ENTRYPOINT
# ---------------------------

async def main():
    bot = Bot(BOT_TOKEN, default=BOT_DEFAULT)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())