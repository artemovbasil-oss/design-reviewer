# bot.py
# Design Reviewer Telegram bot (EN default + RU toggle)
# - Accepts screenshots OR public Figma frame links
# - Retro ASCII progress animation + Cancel button
# - Sends: (1) What I see (2) Verdict + recs + score (3) Annotated screenshot + ASCII concept (monospace)
# - Menu keyboard appears only after /start and at the end of each review
#
# ENV required:
#   BOT_TOKEN=...
#   OPENAI_API_KEY=...
#
# Optional:
#   LLM_MODEL=gpt-4o-mini   (default)
#   MAX_IMAGE_SIDE=1600     (default)
#   ASCII_CONCEPT_COLS=34   (safe mobile width for monospace inside the box)

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import html as html_stdlib

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.exceptions import TelegramBadRequest

import aiohttp
from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1600"))
ASCII_CONCEPT_COLS = int(os.getenv("ASCII_CONCEPT_COLS", "34"))  # inner width (mobile-safe)

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN environment variable")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------
# Helpers: i18n
# -----------------------------
I18N = {
    "en": {
        "start_title": "Design Reviewer is online.",
        "start_body": (
            "Send me:\n"
            "• a screenshot (image)\n"
            "• a public Figma frame link (must be accessible)\n\n"
            "I’ll nitpick your UI/UX and copy like a senior teammate — fair, direct, and useful."
        ),
        "btn_submit": "Submit for review",
        "btn_how": "How it works?",
        "btn_channel": "Product design channel",
        "btn_toggle_to_ru": "RU",
        "btn_toggle_to_en": "EN",
        "btn_cancel": "Cancel",
        "how": "1) Send a screenshot or a public Figma frame link\n2) Watch the retro progress\n3) Get: what I see, verdict+recs, annotations+ASCII concept",
        "ask_send": "Send a screenshot or a public Figma link.",
        "downloading_figma": "Fetching Figma preview…",
        "bad_figma": (
            "Couldn’t fetch a Figma preview.\n"
            "Make sure the file is public and the link contains a valid node-id."
        ),
        "processing": "Review in progress…",
        "cancelled": "Cancelled. Send another screenshot/link when ready.",
        "too_big": "Image is too large. Try a smaller screenshot.",
        "llm_error": "LLM error. Try again (or send a clearer / larger screenshot).",
        "done_footer": "Review finished. Want another one?",
        "channel_msg": "Subscribe here: @prodooktovy",
        "channel_btn": "Open @prodooktovy",
        "score": "Score",
        "what_i_see": "What I see",
        "verdict": "Verdict & recommendations",
        "annotations": "Annotations",
        "concept": "ASCII concept",
        "synth": "Synthesizing…",
    },
    "ru": {
        "start_title": "Design Reviewer на связи.",
        "start_body": (
            "Кидай мне:\n"
            "• скриншот (картинку)\n"
            "• ссылку на публичный фрейм Figma (если файл доступен)\n\n"
            "Я буду докапываться как старший товарищ: честно, по делу, с предложениями как улучшить."
        ),
        "btn_submit": "Закинуть на ревью",
        "btn_how": "Как это работает?",
        "btn_channel": "Канал о продуктовом дизайне",
        "btn_toggle_to_ru": "RU",
        "btn_toggle_to_en": "EN",
        "btn_cancel": "Отмена",
        "how": "1) Отправь скрин или публичную ссылку на фрейм Figma\n2) Посмотри ретро-прогресс\n3) Получи: что вижу, вердикт+рекомендации, аннотации+ASCII концепт",
        "ask_send": "Отправь скриншот или публичную ссылку на Figma.",
        "downloading_figma": "Тяну превью из Figma…",
        "bad_figma": (
            "Не смог скачать превью из Figma.\n"
            "Проверь что файл публичный и в ссылке есть корректный node-id."
        ),
        "processing": "Делаю ревью…",
        "cancelled": "Отменил. Готов — кидай следующий скрин/ссылку.",
        "too_big": "Картинка слишком большая. Попробуй скрин поменьше.",
        "llm_error": "Ошибка LLM. Попробуй ещё раз (или пришли скрин крупнее/четче).",
        "done_footer": "Готово. Ещё один экран?",
        "channel_msg": "Подписаться: @prodooktovy",
        "channel_btn": "Открыть @prodooktovy",
        "score": "Оценка",
        "what_i_see": "Что я вижу",
        "verdict": "Вердикт и рекомендации",
        "annotations": "Аннотации",
        "concept": "ASCII концепт",
        "synth": "Собираю концепт…",
    },
}

DEFAULT_LANG = "en"

CHAT_LANG: Dict[int, str] = {}
RUNNING_TASK: Dict[int, asyncio.Task] = {}
CANCEL_EVENT: Dict[int, asyncio.Event] = {}


def t(chat_id: int, key: str) -> str:
    lang = CHAT_LANG.get(chat_id, DEFAULT_LANG)
    return I18N.get(lang, I18N[DEFAULT_LANG]).get(key, key)


# -----------------------------
# Telegram formatting helpers (HTML)
# -----------------------------
def h(s: str) -> str:
    return html_stdlib.escape(s or "", quote=False)


def html_title(text: str) -> str:
    return f"<b>{h(text)}</b>"


def html_block(title: str, body: str) -> str:
    return f"{html_title(title)}\n{h(body)}"


def html_pre(text: str) -> str:
    return f"<pre>{h(text)}</pre>"


# -----------------------------
# UI: keyboards
# -----------------------------
def main_menu_kb(chat_id: int) -> ReplyKeyboardMarkup:
    lang = CHAT_LANG.get(chat_id, DEFAULT_LANG)
    if lang == "ru":
        submit = I18N["ru"]["btn_submit"]
        how = I18N["ru"]["btn_how"]
        channel = I18N["ru"]["btn_channel"]
        toggle = "🌐 EN"
    else:
        submit = I18N["en"]["btn_submit"]
        how = I18N["en"]["btn_how"]
        channel = I18N["en"]["btn_channel"]
        toggle = "🌐 RU"

    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=submit)],
            [KeyboardButton(text=how)],
            [KeyboardButton(text=channel), KeyboardButton(text=toggle)],  # one row
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        input_field_placeholder="Send a screenshot or a public Figma link…",
    )


def cancel_inline_kb(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t(chat_id, "btn_cancel"), callback_data="cancel_review")]
        ]
    )


def channel_inline_kb(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t(chat_id, "channel_btn"), url="https://t.me/prodooktovy")]
        ]
    )


# -----------------------------
# Retro ASCII animation (restored style)
# - compact frame, edited in-place (no spam)
# -----------------------------
SPINNER = ["|", "/", "-", "\\"]


def _pad(s: str, w: int) -> str:
    if len(s) >= w:
        return s[:w]
    return s + (" " * (w - len(s)))


def retro_frame(step: int, title: str, inner_w: int = 28) -> str:
    # inner_w is content width (not counting borders)
    spin = SPINNER[step % len(SPINNER)]
    scan = (step * 3) % inner_w
    bar = ["-"] * inner_w
    bar[scan] = "#"
    # subtle "trail"
    if scan - 1 >= 0:
        bar[scan - 1] = "="
    if scan + 1 < inner_w:
        bar[scan + 1] = "="

    stats = f"scan:{step%100:02d}  load:{(step*7)%100:02d}  synth:{(step*13)%100:02d}"
    top = "+" + ("-" * inner_w) + "+"
    mid1 = "|" + _pad(title, inner_w) + "|"
    mid2 = "|" + "".join(bar) + f" {spin}"  # note: may exceed by 2
    # force exact width: we clip/pad to borders
    mid2 = "|" + _pad("".join(bar) + f" {spin}", inner_w) + "|"
    mid3 = "|" + _pad(stats, inner_w) + "|"
    bot = "+" + ("-" * inner_w) + "+"

    return "\n".join([top, mid1, mid2, mid3, bot])


async def animate_progress(anchor: Message, chat_id: int, title: str, seconds: float = 3.5) -> None:
    start = time.time()
    msg = await anchor.answer(html_pre(retro_frame(0, title)), reply_markup=cancel_inline_kb(chat_id))
    step = 1
    while time.time() - start < seconds:
        if CANCEL_EVENT.get(chat_id) and CANCEL_EVENT[chat_id].is_set():
            break
        try:
            await msg.edit_text(html_pre(retro_frame(step, title)), reply_markup=cancel_inline_kb(chat_id))
        except TelegramBadRequest:
            break
        await asyncio.sleep(0.25)
        step += 1

    try:
        await msg.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass


# -----------------------------
# Figma preview fetching (no caching)
# -----------------------------
FIGMA_URL_RE = re.compile(r"https?://(www\.)?figma\.com/(file|design)/[^ ]+")


async def fetch_json(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
        r.raise_for_status()
        return await r.json()


async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=25)) as r:
        r.raise_for_status()
        return await r.read()


async def figma_oembed_preview_bytes(figma_url: str) -> Optional[bytes]:
    oembed = f"https://www.figma.com/api/oembed?url={aiohttp.helpers.quote(figma_url, safe='')}"
    async with aiohttp.ClientSession() as session:
        try:
            data = await fetch_json(session, oembed)
        except Exception:
            return None

        thumb = data.get("thumbnail_url") or data.get("thumbnailUrl")
        if not thumb:
            return None

        # bust cache
        sep = "&" if "?" in thumb else "?"
        thumb = f"{thumb}{sep}_ts={int(time.time())}"

        try:
            return await fetch_bytes(session, thumb)
        except Exception:
            return None


# -----------------------------
# Image utils
# -----------------------------
def downscale_image_bytes(img_bytes: bytes, max_side: int) -> bytes:
    im = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h0 = im.size
    m = max(w, h0)
    if m <= max_side:
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h0 * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def to_data_url_png(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# -----------------------------
# ASCII concept formatting (prevents mobile wrapping)
# - rebuilds into a clean box with fixed width (ASCII_CONCEPT_COLS)
# -----------------------------
_BORDER_RE = re.compile(r"^\s*[\+\-]{3,}\s*$")


def _strip_box_art(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    for ln in lines:
        raw = ln.rstrip("\n")
        if not raw.strip():
            cleaned.append("")
            continue
        if _BORDER_RE.match(raw.strip()):
            continue
        s = raw.strip("\r")
        # remove outer pipes if present
        s2 = s
        # left pipe
        if s2.lstrip().startswith("|"):
            i = s2.find("|")
            s2 = s2[i + 1 :]
            if s2.startswith(" "):
                s2 = s2[1:]
        # right pipe
        s2 = s2.rstrip()
        if s2.endswith("|"):
            s2 = s2[:-1].rstrip()
        # drop weird "corner" lines like "+-----+"
        if s2.strip().startswith("+") and set(s2.strip()) <= set("+-+"):
            continue
        cleaned.append(s2.rstrip())
    # trim leading/trailing empty lines
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned


def _wrap_line(s: str, width: int) -> List[str]:
    s = s.replace("\t", "    ")
    if len(s) <= width:
        return [s]
    out: List[str] = []
    cur = s
    while len(cur) > width:
        chunk = cur[:width]
        # try break at last space
        cut = chunk.rfind(" ")
        if cut >= int(width * 0.55):
            out.append(cur[:cut].rstrip())
            cur = cur[cut + 1 :].lstrip()
        else:
            out.append(chunk)
            cur = cur[width:]
    if cur:
        out.append(cur)
    return out


def format_ascii_concept(concept: str, title: str, inner_w: int) -> str:
    if not concept:
        concept = ""
    # normalize line endings
    lines = concept.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = _strip_box_art(lines)

    # wrap to inner width
    wrapped: List[str] = []
    for ln in lines:
        if ln == "":
            wrapped.append("")
            continue
        wrapped.extend(_wrap_line(ln, inner_w))

    # limit height a bit
    wrapped = wrapped[:40] if wrapped else [""]

    top = "+" + ("-" * inner_w) + "+"
    # title line inside box (no emoji / monochrome vibe)
    ttl = _pad(title, inner_w)
    header = "|" + ttl + "|"
    sep = "|" + ("-" * inner_w) + "|"

    body: List[str] = []
    for ln in wrapped:
        body.append("|" + _pad(ln, inner_w) + "|")

    bot = "+" + ("-" * inner_w) + "+"
    return "\n".join([top, header, sep, *body, bot])


# -----------------------------
# LLM calls (Vision)
# -----------------------------
@dataclass
class IssueBox:
    id: str
    title: str
    problem: str
    fix: str
    kind: str  # "ux"|"copy"
    x: float
    y: float
    w: float
    h: float


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    a = text.find("{")
    b = text.rfind("}")
    if a != -1 and b != -1 and b > a:
        chunk = text[a : b + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None


def llm_review(image_data_url: str, lang: str) -> Dict[str, Any]:
    if lang not in ("en", "ru"):
        lang = "en"

    system = (
        "You are a strict senior product designer doing a partner-style design review. "
        "Be direct and critical but not rude; no profanity. "
        "If something is good, explicitly praise what is good and why. "
        "Only GUESS font family vibes and palette vibes; do NOT mention exact sizes, pixels, hex codes, medians. "
        "Return ONLY valid JSON (no markdown)."
    )

    if lang == "ru":
        user_instr = (
            "Сделай ревью по скриншоту интерфейса.\n"
            "Нужно:\n"
            "1) Коротко и ясно описать что ты видишь.\n"
            "2) Дать оценку от 1 до 10.\n"
            "3) Дать единый вердикт и рекомендации (UX+визуал+текст), конкретно: что не так и что сделать.\n"
            "4) Дай список проблем (issues) с боксами по изображению (x,y,w,h нормализованы 0..1).\n"
            "   issue: id, kind ('ux'|'copy'), title, problem, fix, x,y,w,h.\n"
            "5) ASCII-концепт улучшенного экрана (моноширинно, без цветных эмодзи).\n\n"
            "Формат ответа JSON (только эти ключи):\n"
            "{\n"
            '  "what_i_see": "...",\n'
            '  "score": 0,\n'
            '  "verdict": "...",\n'
            '  "issues": [ { "id":"1", "kind":"ux", "title":"...", "problem":"...", "fix":"...", "x":0, "y":0, "w":0, "h":0 } ],\n'
            '  "ascii_concept": "..." \n'
            "}\n"
        )
    else:
        user_instr = (
            "Do a design review of this UI screenshot.\n"
            "You must:\n"
            "1) Describe what you see, short and clear.\n"
            "2) Give a score 1..10.\n"
            "3) ONE combined verdict + recommendations (UX + visuals + copy). Be concrete.\n"
            "4) issues with normalized boxes (x,y,w,h in 0..1).\n"
            "5) ASCII concept of an improved screen (monospace, no colorful emoji).\n\n"
            "Output ONLY valid JSON with exactly these keys:\n"
            "{\n"
            '  "what_i_see": "...",\n'
            '  "score": 0,\n'
            '  "verdict": "...",\n'
            '  "issues": [ { "id":"1", "kind":"ux", "title":"...", "problem":"...", "fix":"...", "x":0, "y":0, "w":0, "h":0 } ],\n'
            '  "ascii_concept": "..." \n'
            "}\n"
        )

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_instr},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            },
        ],
    )

    out_text_parts: List[str] = []
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out_text_parts.append(c.text)

    raw = "\n".join(out_text_parts).strip()
    data = parse_llm_json(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM returned invalid JSON")

    score = data.get("score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0
    score = max(1, min(10, score))

    issues_in = data.get("issues") or []
    issues_out: List[Dict[str, Any]] = []
    if isinstance(issues_in, list):
        for it in issues_in[:20]:
            if not isinstance(it, dict):
                continue
            kind = str(it.get("kind", "ux")).strip().lower()
            if kind not in ("ux", "copy"):
                kind = "ux"
            x = clamp01(safe_float(it.get("x", 0)))
            y = clamp01(safe_float(it.get("y", 0)))
            w = clamp01(safe_float(it.get("w", 0)))
            h0 = clamp01(safe_float(it.get("h", 0)))
            issues_out.append(
                {
                    "id": str(it.get("id", "")).strip() or str(len(issues_out) + 1),
                    "kind": kind,
                    "title": str(it.get("title", "")).strip(),
                    "problem": str(it.get("problem", "")).strip(),
                    "fix": str(it.get("fix", "")).strip(),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h0,
                }
            )

    return {
        "what_i_see": str(data.get("what_i_see", "")).strip(),
        "score": score,
        "verdict": str(data.get("verdict", "")).strip(),
        "issues": issues_out,
        "ascii_concept": str(data.get("ascii_concept", "")).strip(),
    }


# -----------------------------
# Annotation rendering
# -----------------------------
def draw_annotations(img_bytes: bytes, issues: List[Dict[str, Any]]) -> bytes:
    im = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h0 = im.size
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def rect_from_norm(it: Dict[str, Any]) -> Tuple[int, int, int, int]:
        x = int(clamp01(it.get("x", 0.0)) * w)
        y = int(clamp01(it.get("y", 0.0)) * h0)
        ww = int(clamp01(it.get("w", 0.0)) * w)
        hh = int(clamp01(it.get("h", 0.0)) * h0)
        ww = max(14, ww)
        hh = max(14, hh)
        x2 = min(w - 1, x + ww)
        y2 = min(h0 - 1, y + hh)
        return x, y, x2, y2

    # monochrome annotations
    for idx, it in enumerate(issues[:20], start=1):
        x1, y1, x2, y2 = rect_from_norm(it)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=4)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)

        label = str(it.get("id") or idx)[:3]
        lx1, ly1 = x1 + 4, max(0, y1 - 18)
        lx2, ly2 = lx1 + 30, ly1 + 16
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(0, 0, 0))
        if font:
            draw.text((lx1 + 8, ly1 + 2), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((lx1 + 8, ly1 + 2), label, fill=(255, 255, 255))

    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def issues_to_text(chat_id: int, issues: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for it in issues[:12]:
        kid = it.get("id", "")
        kind = it.get("kind", "ux")
        title = (it.get("title", "") or "").strip()
        problem = (it.get("problem", "") or "").strip()
        fix = (it.get("fix", "") or "").strip()
        if not (title or problem or fix):
            continue
        tag = "UX" if kind == "ux" else "COPY"
        lines.append(f"{kid}. [{tag}] {title}".strip())
        if problem:
            lines.append(f"   - {problem}")
        if fix:
            lines.append(f"   - Do: {fix}" if CHAT_LANG.get(chat_id, DEFAULT_LANG) == "en" else f"   - Сделай: {fix}")
    return "\n".join(lines).strip()


# -----------------------------
# Cancel handling
# -----------------------------
async def cancel_current(chat_id: int) -> None:
    ev = CANCEL_EVENT.get(chat_id)
    if ev:
        ev.set()
    task = RUNNING_TASK.get(chat_id)
    if task and not task.done():
        task.cancel()


# -----------------------------
# Core processing
# -----------------------------
async def process_and_reply(anchor: Message, chat_id: int, img_bytes: bytes) -> None:
    CANCEL_EVENT[chat_id] = asyncio.Event()

    # Hide menu during review
    try:
        await anchor.answer(t(chat_id, "processing"), reply_markup=ReplyKeyboardRemove())
    except TelegramBadRequest:
        pass

    # Progress #1
    await animate_progress(anchor, chat_id, title=t(chat_id, "processing"), seconds=3.2)

    if len(img_bytes) > 25 * 1024 * 1024:
        await anchor.answer(t(chat_id, "too_big"), reply_markup=main_menu_kb(chat_id))
        return

    img_bytes_small = downscale_image_bytes(img_bytes, MAX_IMAGE_SIDE)
    img_data_url = to_data_url_png(img_bytes_small)

    if CANCEL_EVENT[chat_id].is_set():
        await anchor.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))
        return

    try:
        data = await asyncio.to_thread(llm_review, img_data_url, CHAT_LANG.get(chat_id, DEFAULT_LANG))
    except asyncio.CancelledError:
        await anchor.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))
        return
    except Exception:
        await anchor.answer(t(chat_id, "llm_error"), reply_markup=main_menu_kb(chat_id))
        return

    if CANCEL_EVENT[chat_id].is_set():
        await anchor.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))
        return

    # 1) What I see
    what_i_see = (data.get("what_i_see", "") or "").strip() or "(No description returned)"
    await anchor.answer(f"{html_title(t(chat_id, 'what_i_see'))}\n{h(what_i_see)}")

    # 2) Verdict + score + list
    score = data.get("score", 0)
    verdict = (data.get("verdict", "") or "").strip()
    issues = data.get("issues", []) or []
    issues_text = issues_to_text(chat_id, issues)

    msg2_parts = [
        html_title(t(chat_id, "verdict")),
        f"{html_title(t(chat_id, 'score'))}: {h(str(score))}/10",
        "",
        h(verdict),
    ]
    if issues_text:
        msg2_parts += ["", html_title(t(chat_id, "annotations")), h(issues_text)]
    await anchor.answer("\n".join(msg2_parts).strip())

    # Progress #2
    await animate_progress(anchor, chat_id, title=t(chat_id, "synth"), seconds=2.6)

    # 3) Annotated image + ASCII concept (fixed width, no wrapping by Telegram)
    try:
        annotated = await asyncio.to_thread(draw_annotations, img_bytes_small, issues)
        await anchor.answer_photo(
            photo=annotated,
            caption=f"{t(chat_id,'annotations')}: {min(len(issues), 20)}",
        )
    except Exception:
        pass

    raw_concept = (data.get("ascii_concept", "") or "").strip()
    boxed = format_ascii_concept(
        raw_concept,
        title=t(chat_id, "concept"),
        inner_w=ASCII_CONCEPT_COLS,
    )
    await anchor.answer(html_pre(boxed))

    # End menu
    await anchor.answer(t(chat_id, "done_footer"), reply_markup=main_menu_kb(chat_id))


# -----------------------------
# Router / Handlers
# -----------------------------
router = Router()


@router.message(Command("start"))
async def on_start(m: Message) -> None:
    chat_id = m.chat.id
    CHAT_LANG.setdefault(chat_id, DEFAULT_LANG)
    await m.answer(
        f"{html_title(t(chat_id,'start_title'))}\n{h(t(chat_id,'start_body'))}",
        reply_markup=main_menu_kb(chat_id),
    )


@router.message(F.text)
async def on_text(m: Message) -> None:
    chat_id = m.chat.id
    text = (m.text or "").strip()

    # Language toggle
    if text in ("🌐 RU", "🌐 EN", "RU", "EN"):
        if text.endswith("RU"):
            CHAT_LANG[chat_id] = "ru"
        else:
            CHAT_LANG[chat_id] = "en"
        await m.answer(f"{html_title('OK')}\n{h(t(chat_id,'ask_send'))}", reply_markup=main_menu_kb(chat_id))
        return

    # Menu buttons
    if text == I18N["en"]["btn_how"] or text == I18N["ru"]["btn_how"]:
        title = "How it works" if CHAT_LANG.get(chat_id) == "en" else "Как это работает"
        await m.answer(html_block(title, t(chat_id, "how")))
        return

    if text == I18N["en"]["btn_channel"] or text == I18N["ru"]["btn_channel"]:
        await m.answer(t(chat_id, "channel_msg"), reply_markup=channel_inline_kb(chat_id))
        return

    if text == I18N["en"]["btn_submit"] or text == I18N["ru"]["btn_submit"]:
        await m.answer(t(chat_id, "ask_send"))
        return

    # Figma link
    if FIGMA_URL_RE.search(text):
        await cancel_current(chat_id)

        async def _run() -> None:
            await m.answer(t(chat_id, "downloading_figma"), reply_markup=ReplyKeyboardRemove())
            await animate_progress(m, chat_id, title=t(chat_id, "downloading_figma"), seconds=2.2)

            if CANCEL_EVENT.get(chat_id) and CANCEL_EVENT[chat_id].is_set():
                await m.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))
                return

            preview = await figma_oembed_preview_bytes(text)
            if not preview:
                await m.answer(t(chat_id, "bad_figma"), reply_markup=main_menu_kb(chat_id))
                return

            # show preview so user sees it's the correct node
            try:
                await m.answer_photo(photo=preview, caption="Figma preview")
            except Exception:
                pass

            await process_and_reply(m, chat_id, preview)

        task = asyncio.create_task(_run())
        RUNNING_TASK[chat_id] = task
        return

    await m.answer(t(chat_id, "ask_send"))


@router.message(F.photo)
async def on_photo(m: Message) -> None:
    chat_id = m.chat.id
    CHAT_LANG.setdefault(chat_id, DEFAULT_LANG)

    await cancel_current(chat_id)

    async def _run() -> None:
        photo = m.photo[-1]
        file = await m.bot.get_file(photo.file_id)
        stream = await m.bot.download_file(file.file_path)
        raw = stream.read() if hasattr(stream, "read") else bytes(stream)
        await process_and_reply(m, chat_id, raw)

    task = asyncio.create_task(_run())
    RUNNING_TASK[chat_id] = task


@router.callback_query(F.data == "cancel_review")
async def on_cancel(cb) -> None:
    chat_id = cb.message.chat.id
    await cancel_current(chat_id)
    try:
        await cb.answer("Cancelled", show_alert=False)
    except Exception:
        pass
    try:
        await cb.message.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))
    except Exception:
        pass


# -----------------------------
# Main
# -----------------------------
async def main() -> None:
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())