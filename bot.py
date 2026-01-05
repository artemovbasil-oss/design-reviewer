#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Design Reviewer Telegram bot (aiogram 3.7+)
- Accepts: screenshots (photo / image document) and public Figma frame links
- Outputs 4 messages per review:
  1) What I see
  2) Verdict + recommendations (UX + Text)
  3) Annotated screenshot (OCR text blocks numbered)
  4) ASCII concept (monospace, fixed width — no ugly wrapping on phone)
- Has:
  - Retro ASCII progress animation (compact)
  - Cancel button during processing
  - Main menu only after /start and at the end of each review
  - Language toggle EN/RU (EN default)
  - Channel button @prodooktovy
Notes:
- No python-dotenv dependency. Uses environment variables only.
- Uses OpenAI Responses API via aiohttp (no httpx/requests requirement).
- OCR uses pytesseract if installed; if not, bot still works (no annotations).
"""

import asyncio
import base64
import io
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from aiogram.client.default import DefaultBotProperties


# ---------------------------
# Config (ENV)
# ---------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN environment variable")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y", "on")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

OCR_LANG = os.getenv("OCR_LANG", "rus+eng").strip()

# LLM robustness
OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "40"))
MAX_IMAGE_W_LLM = int(os.getenv("MAX_IMAGE_W_LLM", "1600"))
ASCII_WIDTH = int(os.getenv("ASCII_WIDTH", "34"))  # tuned for phone width

# Telegram limits: keep progress message compact
PROGRESS_EDIT_INTERVAL = 0.35


# ---------------------------
# Optional OCR
# ---------------------------

TESSERACT_AVAILABLE = True
try:
    import pytesseract  # type: ignore
except Exception:
    TESSERACT_AVAILABLE = False


# ---------------------------
# Simple per-user state
# ---------------------------

USER_LANG: Dict[int, str] = {}        # "EN" / "RU"
RUNNING_TASK: Dict[int, asyncio.Task] = {}  # chat_id -> task


def lang_of(chat_id: int) -> str:
    return USER_LANG.get(chat_id, "EN")


def t(chat_id: int, en: str, ru: str) -> str:
    return en if lang_of(chat_id) == "EN" else ru


# ---------------------------
# Safe HTML helpers
# ---------------------------

import html as py_html


def h(text: str) -> str:
    """Escape for Telegram HTML parse mode."""
    return py_html.escape(text, quote=False)


# ---------------------------
# Keyboards
# ---------------------------

def main_menu_kb(chat_id: int) -> ReplyKeyboardMarkup:
    # Want last two buttons in one row if possible.
    # Telegram ReplyKeyboard rows are arrays.
    review = t(chat_id, "Send for review", "Закинуть на ревью")
    how = t(chat_id, "How it works?", "Как это работает?")
    channel = t(chat_id, "Channel: @prodooktovy", "Канал: @prodooktovy")
    toggle = t(chat_id, "Language: EN/RU", "Язык: EN/RU")
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=review)],
            [KeyboardButton(text=how)],
            [KeyboardButton(text=channel), KeyboardButton(text=toggle)],
        ],
        resize_keyboard=True,
        selective=True,
        input_field_placeholder=t(chat_id, "Drop a screenshot or a public Figma link…", "Кинь скрин или публичную ссылку Figma…"),
    )


def cancel_inline_kb(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t(chat_id, "Cancel", "Отмена"), callback_data="cancel")]
        ]
    )


# ---------------------------
# ASCII animation (compact + retro)
# ---------------------------

FRAMES = [
    r"┌──────────────────┐
│  SCANNING  .      │
│  [=         ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  ..     │
│  [==        ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  ...    │
│  [===       ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  ....   │
│  [====      ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  .....  │
│  [=====     ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  ...... │
│  [======    ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  ...... │
│  [=======   ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  .....  │
│  [========  ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  ....   │
│  [========= ]     │
└──────────────────┘",
    r"┌──────────────────┐
│  SCANNING  ...    │
│  [==========]     │
└──────────────────┘",
]


async def animate_progress(anchor: Message, title: str, chat_id: int) -> Message:
    """
    Sends a compact progress message and edits it with ASCII frames.
    Returns the progress message (to edit or delete later).
    """
    msg = await anchor.answer(
        f"{h(title)}\n<pre>{h(FRAMES[0])}</pre>",
        reply_markup=cancel_inline_kb(chat_id),
        disable_web_page_preview=True,
    )

    async def _runner():
        i = 0
        last = 0.0
        while True:
            now = time.time()
            if now - last < PROGRESS_EDIT_INTERVAL:
                await asyncio.sleep(0.05)
                continue
            last = now
            frame = FRAMES[i % len(FRAMES)]
            i += 1
            try:
                await msg.edit_text(
                    f"{h(title)}\n<pre>{h(frame)}</pre>",
                    reply_markup=cancel_inline_kb(chat_id),
                    disable_web_page_preview=True,
                )
            except Exception:
                # Telegram sometimes rejects edits; ignore and continue
                pass
            await asyncio.sleep(PROGRESS_EDIT_INTERVAL)

    task = asyncio.create_task(_runner())
    RUNNING_TASK[chat_id] = task
    return msg


def stop_progress(chat_id: int) -> None:
    task = RUNNING_TASK.pop(chat_id, None)
    if task and not task.done():
        task.cancel()


# ---------------------------
# Utils: image handling
# ---------------------------

def downscale_for_llm(img_bytes: bytes, max_w: int) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h_ = img.size
    if w <= max_w:
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    new_h = int(h_ * (max_w / w))
    img = img.resize((max_w, new_h), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def img_bytes_to_b64_png(img_bytes: bytes) -> str:
    # Ensure PNG for consistent decoding.
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode("utf-8")


# ---------------------------
# Figma link handling (public oEmbed)
# ---------------------------

FIGMA_URL_RE = re.compile(r"(https?://www\.figma\.com/(file|design)/[^\s]+)", re.IGNORECASE)


async def fetch_figma_preview_image(figma_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Fetches thumbnail via Figma oEmbed (public files).
    Returns (image_bytes, title) or (None, None).
    """
    # Cache-bust to avoid "same result for every link" issues
    # (and to avoid accidental reuse of previous thumbnail).
    busted_url = figma_url
    sep = "&" if "?" in figma_url else "?"
    busted_url = f"{figma_url}{sep}ts={int(time.time()*1000)}"

    oembed = "https://www.figma.com/oembed?url=" + urllib.parse.quote(busted_url, safe="")
    timeout = aiohttp.ClientTimeout(total=20)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(oembed) as r:
            if r.status != 200:
                return None, None
            data = await r.json()
            thumb = data.get("thumbnail_url")
            title = data.get("title")
            if not thumb:
                return None, title

        # Download thumbnail
        async with session.get(thumb) as r2:
            if r2.status != 200:
                return None, title
            return await r2.read(), title


# ---------------------------
# OCR & annotation (numbers on text blocks)
# ---------------------------

@dataclass
class TextBox:
    idx: int
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: int


def ocr_extract_boxes(img_bytes: bytes, lang: str) -> List[TextBox]:
    if not TESSERACT_AVAILABLE:
        return []

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Tesseract gives boxes in the current image pixel space; we draw on the same image => no shifting.
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)  # type: ignore
    n = len(data.get("text", []))
    boxes: List[TextBox] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1

        # filter noisy boxes
        if conf < 55:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h_ = int(data["height"][i])
        if w <= 6 or h_ <= 6:
            continue
        # Avoid random punctuation-only
        if len(re.sub(r"[\W_]+", "", txt, flags=re.UNICODE)) == 0:
            continue

        boxes.append(TextBox(idx=len(boxes) + 1, text=txt, x=x, y=y, w=w, h=h_, conf=conf))

    # sort top-to-bottom left-to-right to stabilize numbering
    boxes.sort(key=lambda b: (b.y, b.x))
    # re-number
    for j, b in enumerate(boxes, start=1):
        b.idx = j
    return boxes


def draw_annotations(img_bytes: bytes, boxes: List[TextBox]) -> Optional[bytes]:
    if not boxes:
        return None

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Use built-in default font (works everywhere)
    font = ImageFont.load_default()

    for b in boxes[:40]:  # don’t spam
        # Black frame (clean, not "red screaming")
        x1, y1 = b.x, b.y
        x2, y2 = b.x + b.w, b.y + b.h
        for t_ in range(2):
            draw.rectangle([x1 - t_, y1 - t_, x2 + t_, y2 + t_], outline=(0, 0, 0))

        label = str(b.idx)
        # small white label background
        tw, th = draw.textsize(label, font=font)
        pad = 2
        draw.rectangle([x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=(255, 255, 255))
        draw.text((x1 + pad, y1 - th - pad), label, fill=(0, 0, 0), font=font)

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def join_box_snippets(boxes: List[TextBox], max_items: int = 24) -> str:
    lines = []
    for b in boxes[:max_items]:
        # keep short
        txt = b.text
        if len(txt) > 40:
            txt = txt[:37] + "…"
        lines.append(f"{b.idx}. {txt}")
    return "\n".join(lines)


# ---------------------------
# ASCII concept generator (fallback + post-processing)
# ---------------------------

def wrap_ascii_lines(s: str, width: int) -> str:
    # Keep lines <= width, avoid breaking ASCII box borders too aggressively.
    out_lines = []
    for line in s.splitlines():
        if len(line) <= width:
            out_lines.append(line)
            continue
        # hard wrap
        for i in range(0, len(line), width):
            out_lines.append(line[i:i + width])
    return "\n".join(out_lines)


def fallback_concept_ascii(width: int) -> str:
    # Simple "concept" when image generation service is unavailable.
    # Must fit phone: width ~34 chars
    base = [
        "┌" + "─" * (width - 2) + "┐",
        "│" + "  CLEANER HIERARCHY".ljust(width - 2) + "│",
        "│" + "  • 1 primary action".ljust(width - 2) + "│",
        "│" + "  • fewer competing labels".ljust(width - 2) + "│",
        "│" + "  • consistent spacing".ljust(width - 2) + "│",
        "│" + "  • calm copy, no бюрократия".ljust(width - 2) + "│",
        "│" + "".ljust(width - 2) + "│",
        "│" + "  [ PRIMARY ]".ljust(width - 2) + "│",
        "│" + "  secondary link".ljust(width - 2) + "│",
        "└" + "─" * (width - 2) + "┘",
    ]
    return "\n".join(base)


# ---------------------------
# OpenAI (Responses API via aiohttp)
# ---------------------------

OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"


def build_prompt(chat_id: int, ocr_snippet: str) -> str:
    # Strictly ask for sections with markers; no JSON.
    # Tone: senior buddy, honest, no swearing.
    # No RGB/px stats; only guess font family (broad).
    return t(
        chat_id,
        en=(
            "You are a senior product designer doing a tough, helpful design review.\n"
            "Be honest, a bit strict, but not rude. No profanity.\n"
            "Only use black/white emojis (e.g., ✅ ❌ ⚠️). Keep it practical.\n\n"
            "Rules:\n"
            "- Do NOT output JSON.\n"
            "- Use exactly these 4 sections, with the exact headers:\n"
            "===WHAT_I_SEE===\n"
            "===VERDICT_AND_RECOMMENDATIONS===\n"
            "===TEXT_ISSUES_AND_FIXES===\n"
            "===ASCII_CONCEPT===\n\n"
            "Requirements:\n"
            "- Give an overall score 0–10.\n"
            "- For fonts/palette: only GUESS the font family vibe (e.g., 'looks like Inter/SF/Roboto'), no pixel sizes, no color values.\n"
            "- If something is good, praise it and say what exactly.\n"
            "- If something is bad, call it out clearly and propose fixes.\n"
            "- ASCII_CONCEPT must be a neat monospace block, max line length "
            f"{ASCII_WIDTH} characters.\n\n"
            "OCR snippet (may be incomplete):\n"
            f"{ocr_snippet}\n"
        ),
        ru=(
            "Ты — сеньор продуктовый дизайнер, делаешь жёсткое, но полезное ревью.\n"
            "Честно, строго, но без хамства и без мата.\n"
            "Эмодзи только ч/б (✅ ❌ ⚠️). Практика важнее поэзии.\n\n"
            "Правила:\n"
            "- НЕ выводи JSON.\n"
            "- Ровно 4 секции с такими заголовками:\n"
            "===WHAT_I_SEE===\n"
            "===VERDICT_AND_RECOMMENDATIONS===\n"
            "===TEXT_ISSUES_AND_FIXES===\n"
            "===ASCII_CONCEPT===\n\n"
            "Требования:\n"
            "- Общая оценка 0–10.\n"
            "- Про шрифты/палитру: только ПРЕДПОЛОЖЕНИЕ про семейство (Inter/SF/Roboto и т.п.), без размеров/цветов.\n"
            "- Если хорошо — похвали и скажи, что именно хорошо.\n"
            "- Если плохо — назови косяк прямо и предложи улучшение.\n"
            "- ASCII_CONCEPT — аккуратный моноширинный блок, длина строки максимум "
            f"{ASCII_WIDTH} символов.\n\n"
            "OCR-сниппет (может быть неполным):\n"
            f"{ocr_snippet}\n"
        ),
    )


async def openai_review(chat_id: int, img_bytes: bytes, ocr_snippet: str) -> Tuple[str, str, str, str]:
    """
    Returns 4 sections: what_i_see, verdict, text_issues, ascii_concept
    Raises on errors so caller can fallback.
    """
    if not OPENAI_API_KEY or not LLM_ENABLED:
        raise RuntimeError("LLM disabled or missing key")

    img_b64 = img_bytes_to_b64_png(img_bytes)
    prompt = build_prompt(chat_id, ocr_snippet)

    payload = {
        "model": LLM_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                ],
            }
        ],
        "max_output_tokens": 900,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=OPENAI_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(OPENAI_ENDPOINT, headers=headers, data=json.dumps(payload)) as r:
            txt = await r.text()
            if r.status != 200:
                # Raise with exact reason for logs
                raise RuntimeError(f"OpenAI {r.status}: {txt[:800]}")
            data = json.loads(txt)

    # Responses API: try common shapes
    # We want the final text output.
    out_text = ""
    # 1) output_text convenience
    if isinstance(data, dict) and data.get("output_text"):
        out_text = data["output_text"]
    # 2) output array with content blocks
    if not out_text and isinstance(data, dict):
        out = data.get("output", [])
        parts = []
        for item in out:
            content = item.get("content", []) if isinstance(item, dict) else []
            for c in content:
                if c.get("type") in ("output_text", "summary_text") and "text" in c:
                    parts.append(c["text"])
        out_text = "\n".join(parts).strip()

    if not out_text:
        raise RuntimeError("OpenAI returned empty output")

    # Parse sections
    def pick(section: str) -> str:
        m = re.search(rf"==={re.escape(section)}===\s*(.*?)(?=\n===|$)", out_text, flags=re.S)
        return (m.group(1).strip() if m else "").strip()

    what_i_see = pick("WHAT_I_SEE")
    verdict = pick("VERDICT_AND_RECOMMENDATIONS")
    text_issues = pick("TEXT_ISSUES_AND_FIXES")
    concept = pick("ASCII_CONCEPT")

    # Clean typical "['text']" garbage if model accidentally prints python-ish lists
    def strip_list_repr(s: str) -> str:
        s2 = s.strip()
        if (s2.startswith("['") and s2.endswith("']")) or (s2.startswith('["') and s2.endswith('"]')):
            s2 = s2[2:-2]
        return s2.strip()

    what_i_see = strip_list_repr(what_i_see)
    verdict = strip_list_repr(verdict)
    text_issues = strip_list_repr(text_issues)
    concept = strip_list_repr(concept)

    if not concept:
        concept = fallback_concept_ascii(ASCII_WIDTH)

    # Enforce width to avoid phone wrapping
    concept = wrap_ascii_lines(concept, ASCII_WIDTH)

    # Guard: if model forgot sections, treat as failure -> fallback
    if not (what_i_see and verdict and text_issues):
        raise RuntimeError("LLM output missing sections")

    return what_i_see, verdict, text_issues, concept


# ---------------------------
# Fallback review (OCR-only)
# ---------------------------

def fallback_review(chat_id: int, boxes: List[TextBox]) -> Tuple[str, str, str, str]:
    snippet = join_box_snippets(boxes) if boxes else ""
    what_i_see = t(
        chat_id,
        en="I can’t run full AI review right now. I extracted the visible text blocks and will comment based on that.",
        ru="Сейчас не могу сделать полный AI-разбор. Я вытащил видимые текстовые блоки и дам комментарии по ним.",
    )

    verdict = t(
        chat_id,
        en=(
            "Score: 5/10\n"
            "✅ Good: you shipped something — but it needs cleanup.\n"
            "❌ Issues: unclear hierarchy + too many competing messages.\n"
            "Fix:\n"
            "1) One primary action per screen.\n"
            "2) Shorten labels. Remove bureaucracy.\n"
            "3) Make the next step obvious."
        ),
        ru=(
            "Оценка: 5/10\n"
            "✅ Плюс: экран живой — но его нужно причесать.\n"
            "❌ Минусы: неясная иерархия + много конкурирующих сообщений.\n"
            "Что делать:\n"
            "1) Один главный CTA на экран.\n"
            "2) Сократить формулировки, убрать канцелярит.\n"
            "3) Следующий шаг должен читаться без усилий."
        ),
    )

    text_issues = t(
        chat_id,
        en=("Extracted text blocks:\n" + (snippet or "—")),
        ru=("Вытащенные текстовые блоки:\n" + (snippet or "—")),
    )

    concept = wrap_ascii_lines(fallback_concept_ascii(ASCII_WIDTH), ASCII_WIDTH)
    return what_i_see, verdict, text_issues, concept


# ---------------------------
# Review pipeline
# ---------------------------

async def process_review_from_image(m: Message, img_bytes: bytes) -> None:
    chat_id = m.chat.id

    # Start animation
    title = t(chat_id, "Review in progress…", "Идёт ревью…")
    progress = await animate_progress(m, title=title, chat_id=chat_id)

    try:
        # OCR (for annotations + snippet)
        boxes = ocr_extract_boxes(img_bytes, OCR_LANG) if TESSERACT_AVAILABLE else []
        ocr_snip = join_box_snippets(boxes) if boxes else ""

        # LLM review (downscale for stability)
        what_i_see = verdict = text_issues = concept = ""
        try:
            img_llm = downscale_for_llm(img_bytes, MAX_IMAGE_W_LLM)
            what_i_see, verdict, text_issues, concept = await openai_review(chat_id, img_llm, ocr_snip)
        except Exception as e:
            # Log exact error (Railway)
            print("[LLM ERROR]", type(e).__name__, str(e)[:1200])
            what_i_see, verdict, text_issues, concept = fallback_review(chat_id, boxes)

        # Stop animation
        stop_progress(chat_id)
        try:
            await progress.edit_text(t(chat_id, "Done.", "Готово."), reply_markup=None)
        except Exception:
            pass

        # Message 1: what I see
        await m.answer(h(what_i_see), reply_markup=None, disable_web_page_preview=True)

        # Message 2: verdict + recs + text in one (as requested previously)
        combined = f"{verdict}\n\n{text_issues}".strip()
        await m.answer(h(combined), reply_markup=None, disable_web_page_preview=True)

        # Message 3: annotated screenshot (numbers match OCR blocks)
        annotated = draw_annotations(img_bytes, boxes) if boxes else None
        if annotated:
            caption = t(
                chat_id,
                en="Annotated text blocks (OCR). Numbers match the extracted list.",
                ru="Аннотации текстовых блоков (OCR). Номера совпадают со списком.",
            )
            await m.answer_photo(
                photo=annotated,
                caption=caption,
                reply_markup=None,
            )
        else:
            await m.answer(
                t(chat_id, "No readable text blocks detected for annotation.", "Не нашёл читаемых текстовых блоков для аннотаций."),
                reply_markup=None,
            )

        # Between 3 and 4: small extra animation (keep it short)
        title2 = t(chat_id, "Drafting an ASCII concept…", "Собираю ASCII-концепт…")
        progress2 = await animate_progress(m, title=title2, chat_id=chat_id)
        await asyncio.sleep(1.2)
        stop_progress(chat_id)
        try:
            await progress2.edit_text(t(chat_id, "Concept ready.", "Концепт готов."), reply_markup=None)
        except Exception:
            pass

        # Message 4: ASCII concept (monospace <pre>)
        concept = wrap_ascii_lines(concept, ASCII_WIDTH)
        await m.answer(
            f"<pre>{h(concept)}</pre>",
            reply_markup=None,
            disable_web_page_preview=True,
        )

    finally:
        # Ensure progress stops even on unexpected errors
        stop_progress(chat_id)
        # Show menu only at the end of review
        await m.answer(
            t(chat_id, "Menu:", "Меню:"),
            reply_markup=main_menu_kb(chat_id),
            disable_web_page_preview=True,
        )


async def process_review_from_figma_link(m: Message, url: str) -> None:
    chat_id = m.chat.id

    title = t(chat_id, "Fetching Figma preview…", "Качаю превью из Figma…")
    progress = await animate_progress(m, title=title, chat_id=chat_id)

    try:
        img_bytes, fig_title = await fetch_figma_preview_image(url)
        stop_progress(chat_id)
        try:
            await progress.edit_text(t(chat_id, "Preview fetched.", "Превью скачано."), reply_markup=None)
        except Exception:
            pass

        if not img_bytes:
            await m.answer(
                t(
                    chat_id,
                    "I couldn’t fetch a preview. Make sure the Figma file is public and the link points to a frame/node.",
                    "Не смог скачать превью. Проверь, что файл публичный и ссылка ведёт на фрейм/ноду.",
                ),
                reply_markup=main_menu_kb(chat_id),
            )
            return

        # Show preview image first (user asked)
        cap = t(chat_id, "Figma preview:", "Превью Figma:")
        if fig_title:
            cap += f" {fig_title}"
        await m.answer_photo(photo=img_bytes, caption=cap)

        # Then run review on that preview
        await process_review_from_image(m, img_bytes)

    finally:
        stop_progress(chat_id)


# ---------------------------
# Router handlers
# ---------------------------

dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(m: Message):
    chat_id = m.chat.id
    # EN default
    USER_LANG.setdefault(chat_id, "EN")
    text = t(
        chat_id,
        "I’m your Design Reviewer partner.\n"
        "Send me a screenshot OR a public Figma frame link.\n"
        "I’ll nitpick the UI/UX and the copy (fairly).",
        "Я твой партнёр-дизайн-ревьюер.\n"
        "Кидай скриншот ИЛИ публичную ссылку на фрейм Figma.\n"
        "Я докопаюсь до UI/UX и текста (по делу).",
    )
    await m.answer(h(text), reply_markup=main_menu_kb(chat_id), disable_web_page_preview=True)


@dp.message(Command("menu"))
async def on_menu(m: Message):
    chat_id = m.chat.id
    await m.answer(t(chat_id, "Menu:", "Меню:"), reply_markup=main_menu_kb(chat_id))


@dp.callback_query(F.data == "cancel")
async def on_cancel(cb: CallbackQuery):
    chat_id = cb.message.chat.id if cb.message else cb.from_user.id
    stop_progress(chat_id)
    # Cancel running task if any
    task = RUNNING_TASK.get(chat_id)
    if task and not task.done():
        task.cancel()
    try:
        await cb.answer(t(chat_id, "Cancelled.", "Отменено."), show_alert=False)
    except Exception:
        pass
    if cb.message:
        try:
            await cb.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        await cb.message.answer(t(chat_id, "Stopped. Back to menu.", "Остановил. Возвращаю меню."), reply_markup=main_menu_kb(chat_id))


@dp.message()
async def on_text(m: Message):
    chat_id = m.chat.id
    txt = (m.text or "").strip()

    # Channel button
    if txt == t(chat_id, "Channel: @prodooktovy", "Канал: @prodooktovy"):
        await m.answer("@prodooktovy", disable_web_page_preview=True, reply_markup=main_menu_kb(chat_id))
        return

    # Language toggle
    if txt == t(chat_id, "Language: EN/RU", "Язык: EN/RU"):
        USER_LANG[chat_id] = "RU" if lang_of(chat_id) == "EN" else "EN"
        await m.answer(
            t(chat_id, "Language switched.", "Язык переключён."),
            reply_markup=main_menu_kb(chat_id),
        )
        return

    # How it works
    if txt == t(chat_id, "How it works?", "Как это работает?"):
        await m.answer(
            h(
                t(
                    chat_id,
                    "1) Tap “Send for review” (or just send a screenshot / Figma link)\n"
                    "2) I review what I see + UX + text\n"
                    "3) You get annotations + an ASCII concept",
                    "1) Нажми «Закинуть на ревью» (или просто пришли скрин / ссылку Figma)\n"
                    "2) Я разбираю что вижу + UX + текст\n"
                    "3) Ты получаешь аннотации + ASCII-концепт",
                )
            ),
            reply_markup=main_menu_kb(chat_id),
        )
        return

    # Send for review button (just instruction)
    if txt == t(chat_id, "Send for review", "Закинуть на ревью"):
        await m.answer(
            h(
                t(
                    chat_id,
                    "Send a screenshot (image) or paste a public Figma frame link.",
                    "Пришли скриншот (картинку) или вставь публичную ссылку на фрейм Figma.",
                )
            ),
            reply_markup=None,
        )
        return

    # Figma link
    murl = FIGMA_URL_RE.search(txt)
    if murl:
        url = murl.group(1)
        await process_review_from_figma_link(m, url)
        return

    # Default
    await m.answer(
        h(
            t(
                chat_id,
                "I accept:\n• screenshots (images)\n• public Figma frame links\n\nSend one to start.",
                "Я принимаю:\n• скриншоты (картинки)\n• публичные ссылки на фреймы Figma\n\nПришли — и начнём.",
            )
        ),
        reply_markup=main_menu_kb(chat_id),
        disable_web_page_preview=True,
    )


@dp.message(F.photo)
async def on_photo(m: Message):
    chat_id = m.chat.id

    # Telegram gives multiple sizes; pick the largest
    photo = m.photo[-1]
    file = await m.bot.get_file(photo.file_id)
    img_bytes = await m.bot.download_file(file.file_path)

    await process_review_from_image(m, img_bytes.getvalue() if hasattr(img_bytes, "getvalue") else img_bytes)


@dp.message(F.document)
async def on_document(m: Message):
    chat_id = m.chat.id
    doc = m.document
    if not doc:
        return
    # Allow image documents only
    if not (doc.mime_type or "").startswith("image/"):
        await m.answer(t(chat_id, "Send an image file.", "Пришли картинку."), reply_markup=main_menu_kb(chat_id))
        return

    file = await m.bot.get_file(doc.file_id)
    img_bytes = await m.bot.download_file(file.file_path)

    await process_review_from_image(m, img_bytes.getvalue() if hasattr(img_bytes, "getvalue") else img_bytes)


# ---------------------------
# Entrypoint
# ---------------------------

async def main():
    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    print(f"✅ Bot starting… LLM_ENABLED={LLM_ENABLED}, model={LLM_MODEL}, OCR={TESSERACT_AVAILABLE}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())