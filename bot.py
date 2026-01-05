#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Design Reviewer Telegram Bot (EN default, RU toggle)

What it does:
- Accepts screenshots (photos) OR public Figma frame links.
- Shows retro ASCII progress animation + Cancel button while processing.
- Replies with 4 outputs:
  1) What I see (text)
  2) Verdict + recommendations (UX + copy) + score (text)
  3) Annotated screenshot with issue boxes (image)
  4) ASCII concept (monospace), with a short progress animation before it

Notes:
- No python-dotenv dependency. Uses environment variables.
- Uses OpenAI Responses API via stdlib urllib (no openai/httpx/requests dependency required).
- aiogram v3.x

Required env vars:
- BOT_TOKEN
- OPENAI_API_KEY

Optional env vars:
- LLM_MODEL (default: "gpt-4o-mini")
- LLM_ENABLED (default: "true")
"""

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from html import escape as html_escape
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import Request, urlopen

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.enums import ParseMode

from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Config
# ----------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")

# ASCII constraints for Telegram mobile (monospace <pre> tends to wrap after ~40-45 chars)
ASCII_CONCEPT_COLS = 38  # tuned to avoid wrapping like on your screenshot
MAX_ISSUES = 14

# Retro spinner frames (safe, simple, “no weirdness”)
SPINNER = ["|", "/", "-", "\\"]
BLOCKS = ["░", "▒", "▓", "█"]  # monochrome-ish blocks

# In-memory per-user state
USER_LANG: Dict[int, str] = {}          # chat_id -> "en" | "ru"
RUNNING_TASK: Dict[int, asyncio.Task] = {}  # chat_id -> task


# ----------------------------
# i18n
# ----------------------------

TXT = {
    "en": {
        "welcome_title": "AutoDushnila — Design Reviewer",
        "welcome_body": (
            "Send me a UI screenshot or a public Figma frame link.\n"
            "I’ll be picky (in a useful way): UX + copy + a retro ASCII concept."
        ),
        "menu_drop": "Drop for review",
        "menu_how": "How it works?",
        "menu_channel": "Product design channel",
        "how": (
            "How it works:\n"
            "1) Send a screenshot OR a public Figma frame link.\n"
            "2) I analyze what I see and the UI text.\n"
            "3) You get: description → verdict/score → annotated screenshot → ASCII concept."
        ),
        "need_input": "Send a screenshot or paste a public Figma frame link.",
        "accepted": "Accepted. Uploading…",
        "downloading_figma": "Got the Figma link. Downloading preview…",
        "figma_failed": "Couldn’t fetch a preview from this Figma link. Make sure the file is public.",
        "processing": "Review in progress",
        "thinking": "Thinking",
        "cancel": "Cancel",
        "cancelled": "Cancelled. Send another screenshot/link.",
        "busy": "I’m already reviewing something. Hit Cancel or wait for the report.",
        "llm_off": "LLM is disabled on this deployment. Enable LLM_ENABLED=true.",
        "llm_bad": (
            "I couldn’t parse the review output.\n"
            "Send the same screenshot again (or zoom/crop so the text is readable)."
        ),
        "channel_msg": "Subscribe here: @prodooktovy",
        "done": "Done. Another screen?",
        "score": "Score",
        "what_i_see": "What I see",
        "verdict": "Verdict & recommendations",
        "issues": "Main issues",
        "ascii_concept": "ASCII concept",
    },
    "ru": {
        "welcome_title": "AutoDushnila — Дизайн-ревьюер",
        "welcome_body": (
            "Кидай скрин интерфейса или ссылку на публичный фрейм Figma.\n"
            "Я докопаюсь по делу: UX + текст + ретро ASCII-концепт."
        ),
        "menu_drop": "Закинуть на ревью",
        "menu_how": "Как это работает?",
        "menu_channel": "Канал о продуктовом дизайне",
        "how": (
            "Как это работает:\n"
            "1) Отправь скрин или ссылку на публичный фрейм Figma.\n"
            "2) Я разберу UX и тексты.\n"
            "3) В ответ: что вижу → вердикт/оценка → аннотации → ASCII-концепт."
        ),
        "need_input": "Отправь скриншот или вставь ссылку на публичный фрейм Figma.",
        "accepted": "Принял. Загружаю…",
        "downloading_figma": "Вижу ссылку Figma. Скачиваю превью…",
        "figma_failed": "Не смог скачать превью по ссылке. Проверь, что файл публичный.",
        "processing": "Ревью в процессе",
        "thinking": "Думаю",
        "cancel": "Отмена",
        "cancelled": "Отменил. Кидай следующий скрин/ссылку.",
        "busy": "Я уже занят ревью. Нажми «Отмена» или дождись отчёта.",
        "llm_off": "LLM выключен на этом деплое. Включи LLM_ENABLED=true.",
        "llm_bad": (
            "Не смог распарсить ответ ревью.\n"
            "Пришли тот же скрин ещё раз (или крупнее/обрезанный по делу)."
        ),
        "channel_msg": "Подписывайся: @prodooktovy",
        "done": "Готово. Ещё один экран?",
        "score": "Оценка",
        "what_i_see": "Что я вижу",
        "verdict": "Вердикт и рекомендации",
        "issues": "Главные проблемы",
        "ascii_concept": "ASCII-концепт",
    },
}


def lang_of(chat_id: int) -> str:
    return USER_LANG.get(chat_id, "en")


def t(chat_id: int, key: str) -> str:
    lang = lang_of(chat_id)
    return TXT.get(lang, TXT["en"]).get(key, TXT["en"].get(key, key))


# ----------------------------
# Telegram UI
# ----------------------------

def main_menu_kb(chat_id: int) -> ReplyKeyboardMarkup:
    lang = lang_of(chat_id)
    drop = TXT[lang]["menu_drop"]
    how = TXT[lang]["menu_how"]
    channel = TXT[lang]["menu_channel"]
    lang_btn = "🌐 EN" if lang == "en" else "🌐 RU"

    # Reply keyboard cannot open URLs directly; channel button will trigger a message with an inline URL button.
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=drop)],
            [KeyboardButton(text=how)],
            [KeyboardButton(text=channel), KeyboardButton(text=lang_btn)],
        ],
        resize_keyboard=True,
        input_field_placeholder=TXT[lang]["need_input"],
        selective=False,
    )
    return kb


def cancel_inline_kb(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t(chat_id, "cancel"), callback_data="cancel")]
        ]
    )


def channel_inline_kb() -> InlineKeyboardMarkup:
    # Using telegram deep link
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="@prodooktovy", url="https://t.me/prodooktovy")]
        ]
    )


# ----------------------------
# Helpers
# ----------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def html_pre(s: str) -> str:
    # Wrap as <pre> with HTML escaping; keeps monospace.
    return f"<pre>{html_escape(s)}</pre>"


def pad(s: str, w: int) -> str:
    if len(s) >= w:
        return s[:w]
    return s + (" " * (w - len(s)))


def retro_frame(step: int, title: str, inner_w: int = 34) -> str:
    spin = SPINNER[step % len(SPINNER)]

    bar_w = 12
    pos = step % bar_w
    bar = ["-"] * bar_w
    bar[pos] = "#"
    if pos - 1 >= 0:
        bar[pos - 1] = "="
    if pos + 1 < bar_w:
        bar[pos + 1] = "="
    bar_str = "[" + "".join(bar) + "]"

    scan = f"{(step * 3) % 100:02d}"
    load = f"{(step * 7) % 100:02d}"
    synth = f"{(step * 13) % 100:02d}"

    line1 = f"{title} {bar_str} {spin}"
    line2 = f"scan: {scan}  load: {load}  synth: {synth}"

    top = "+" + ("-" * inner_w) + "+"
    mid1 = "|" + pad(line1, inner_w) + "|"
    mid2 = "|" + pad(line2, inner_w) + "|"
    bot = "+" + ("-" * inner_w) + "+"
    return "\n".join([top, mid1, mid2, bot])


def is_probably_figma_link(text: str) -> bool:
    text = text.strip()
    return "figma.com" in text and ("node-id=" in text or "/design/" in text or "/file/" in text)


def extract_figma_url(text: str) -> Optional[str]:
    # Grab first URL-like token
    m = re.search(r"(https?://\S+)", text.strip())
    if not m:
        return None
    return m.group(1).rstrip(").,]")


def fetch_url_bytes(url: str, timeout: int = 25) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (DesignReviewerBot)",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def figma_oembed(url: str) -> Optional[Dict[str, Any]]:
    # Public oEmbed works for public files
    api = "https://www.figma.com/api/oembed?" + urlencode({"url": url})
    try:
        raw = fetch_url_bytes(api, timeout=25)
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return None


def download_figma_preview(url: str) -> Optional[bytes]:
    data = figma_oembed(url)
    if not data:
        return None
    thumb = data.get("thumbnail_url") or data.get("thumbnail_url_with_viewport")
    if not thumb:
        return None
    try:
        return fetch_url_bytes(thumb, timeout=25)
    except Exception:
        return None


def image_bytes_to_data_url(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    # assume PNG/JPEG; data URL doesn't strictly require exact mime for the API to work
    return "data:image/png;base64," + b64


def parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()

    # If model returned something like {'key': ...} with single quotes -> try to fix
    # Also attempt to extract the first JSON object inside the text.
    candidates: List[str] = []

    # 1) direct
    candidates.append(raw)

    # 2) extract {...} block
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))

    # 3) replace smart quotes / single quotes (best-effort)
    def normalize_quotes(s: str) -> str:
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        # If looks like python dict: replace single quotes with double quotes (risky but helps)
        if "'" in s and '"' not in s:
            s = s.replace("'", '"')
        return s

    for c in candidates:
        c2 = normalize_quotes(c)
        try:
            data = json.loads(c2)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


# ----------------------------
# OpenAI Responses API (stdlib)
# ----------------------------

def openai_responses_create(model: str, input_payload: List[Dict[str, Any]], timeout: int = 60) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/responses"
    body = json.dumps(
        {
            "model": model,
            "input": input_payload,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="ignore")
        return json.loads(raw)


def extract_output_text(resp: Dict[str, Any]) -> str:
    # Responses API returns resp["output"] list with message items, content items
    out_parts: List[str] = []
    for item in resp.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text":
                    out_parts.append(c.get("text", ""))
    return "\n".join(out_parts).strip()


def llm_review_once(image_data_url: str, lang: str, repair_hint: str = "") -> Dict[str, Any]:
    if not LLM_ENABLED:
        raise RuntimeError("LLM_DISABLED")
    if not OPENAI_API_KEY:
        raise RuntimeError("NO_OPENAI_KEY")

    if lang not in ("en", "ru"):
        lang = "en"

    system = (
        "You are a strict senior product designer doing a partner-style design review. "
        "Be direct and critical but not rude; no profanity. "
        "Use only black/white emoji when needed (keep it minimal). "
        "If something is good, explicitly praise what is good and why. "
        "Only GUESS font family vibes and palette vibes; do NOT mention exact sizes, pixels, hex codes. "
        "Return ONLY valid JSON (no markdown, no code fences, no extra text)."
    )

    if lang == "ru":
        user_instr = (
            "Сделай ревью по скриншоту интерфейса.\n"
            "Нужно:\n"
            "1) Коротко описать что ты видишь.\n"
            "2) Оценка от 1 до 10.\n"
            "3) ЕДИНЫЙ вердикт и рекомендации (UX+визуал+текст) — конкретно: что не так и что сделать.\n"
            "4) issues: список проблем с боксами по изображению (x,y,w,h нормализованы 0..1).\n"
            "   issue: id, kind ('ux'|'copy'), title, problem, fix, x,y,w,h.\n"
            "5) ASCII-концепт улучшенного экрана (короткие строки, без длинных токенов).\n\n"
            "Формат ответа JSON (строго эти ключи):\n"
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
            "1) Describe what you see (short and clear).\n"
            "2) Score 1..10.\n"
            "3) ONE combined verdict + recommendations (UX + visuals + copy). Be concrete: what's wrong and what to do.\n"
            "4) issues: list with normalized boxes (x,y,w,h in 0..1).\n"
            "   issue: id, kind ('ux'|'copy'), title, problem, fix, x,y,w,h.\n"
            "5) ASCII concept of an improved screen (short lines; avoid long unbroken tokens).\n\n"
            "Output ONLY valid JSON with exactly these keys:\n"
            "{\n"
            '  "what_i_see": "...",\n'
            '  "score": 0,\n'
            '  "verdict": "...",\n'
            '  "issues": [ { "id":"1", "kind":"ux", "title":"...", "problem":"...", "fix":"...", "x":0, "y":0, "w":0, "h":0 } ],\n'
            '  "ascii_concept": "..." \n'
            "}\n"
        )

    if repair_hint:
        user_instr += "\nIMPORTANT: " + repair_hint

    payload = [
        {"role": "system", "content": [{"type": "input_text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_instr},
                {"type": "input_image", "image_url": image_data_url},
            ],
        },
    ]

    resp = openai_responses_create(LLM_MODEL, payload, timeout=75)
    raw = extract_output_text(resp)
    data = parse_llm_json(raw)
    if not isinstance(data, dict):
        raise ValueError("INVALID_JSON")

    # normalize
    try:
        score = int(data.get("score", 0))
    except Exception:
        score = 0
    score = max(1, min(10, score))

    issues_in = data.get("issues") or []
    issues_out: List[Dict[str, Any]] = []
    if isinstance(issues_in, list):
        for it in issues_in[:MAX_ISSUES]:
            if not isinstance(it, dict):
                continue
            kind = str(it.get("kind", "ux")).strip().lower()
            if kind not in ("ux", "copy"):
                kind = "ux"
            issues_out.append(
                {
                    "id": str(it.get("id", "")).strip() or str(len(issues_out) + 1),
                    "kind": kind,
                    "title": str(it.get("title", "")).strip(),
                    "problem": str(it.get("problem", "")).strip(),
                    "fix": str(it.get("fix", "")).strip(),
                    "x": clamp01(safe_float(it.get("x", 0))),
                    "y": clamp01(safe_float(it.get("y", 0))),
                    "w": clamp01(safe_float(it.get("w", 0))),
                    "h": clamp01(safe_float(it.get("h", 0))),
                }
            )

    return {
        "what_i_see": str(data.get("what_i_see", "")).strip(),
        "score": score,
        "verdict": str(data.get("verdict", "")).strip(),
        "issues": issues_out,
        "ascii_concept": str(data.get("ascii_concept", "")).strip(),
    }


def llm_review(image_data_url: str, lang: str) -> Dict[str, Any]:
    # attempt 1
    try:
        return llm_review_once(image_data_url, lang)
    except Exception:
        # attempt 2 (force JSON-only)
        repair_hint = (
            "Return JSON only. No extra text. No markdown. "
            "Keep only required keys. Escape quotes properly."
        )
        return llm_review_once(image_data_url, lang, repair_hint=repair_hint)


# ----------------------------
# Annotation rendering
# ----------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont:
    # Portable fallback: try DejaVuSans, else PIL default.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def annotate_image(img_bytes: bytes, issues: List[Dict[str, Any]]) -> bytes:
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    font = load_font(18)

    # Draw boxes
    for idx, it in enumerate(issues, start=1):
        x = int(clamp01(it["x"]) * w)
        y = int(clamp01(it["y"]) * h)
        bw = int(clamp01(it["w"]) * w)
        bh = int(clamp01(it["h"]) * h)

        # guard: if model returned nonsense, skip
        if bw < 8 or bh < 8:
            continue

        x2 = min(w - 1, x + bw)
        y2 = min(h - 1, y + bh)

        # monochrome “review” look: white outline + black shadow
        for off in (0, 1):
            draw.rectangle([x - off, y - off, x2 + off, y2 + off], outline=(255, 255, 255))

        # label
        label = str(idx)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad_px = 6
        box = [x, max(0, y - th - 2 * pad_px), x + tw + 2 * pad_px, y]
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((box[0] + pad_px, box[1] + pad_px), label, fill=(255, 255, 255), font=font)

    out = BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ----------------------------
# ASCII concept formatting (NO wrapping)
# ----------------------------

def hard_wrap_lines(text: str, width: int) -> str:
    """
    Wrap preserving words where possible, but also break long tokens.
    Ensures no line exceeds `width`.
    """
    lines_in = (text or "").splitlines() or [""]
    out_lines: List[str] = []

    for line in lines_in:
        s = line.rstrip("\n")
        if not s:
            out_lines.append("")
            continue

        # If the line is already within width, keep
        if len(s) <= width:
            out_lines.append(s)
            continue

        # Word wrap with forced breaks for long tokens
        cur = ""
        for token in re.split(r"(\s+)", s):
            if token.strip() == "":
                # whitespace
                if len(cur) + len(token) <= width:
                    cur += token
                else:
                    out_lines.append(cur.rstrip())
                    cur = ""
                continue

            # token word
            while len(token) > width:
                # flush current
                if cur.strip():
                    out_lines.append(cur.rstrip())
                    cur = ""
                out_lines.append(token[:width])
                token = token[width:]

            if len(cur) + len(token) <= width:
                cur += token
            else:
                out_lines.append(cur.rstrip())
                cur = token

        if cur.strip() or cur == "":
            out_lines.append(cur.rstrip())

    # final guard: trim each line
    out_lines = [ln[:width] for ln in out_lines]
    return "\n".join(out_lines).rstrip()


def ascii_box(title: str, body: str, width: int) -> str:
    body = hard_wrap_lines(body, width - 4)
    body_lines = body.splitlines() if body else [""]
    top = "+" + "-" * (width - 2) + "+"
    ttl = "| " + pad(title, width - 4) + " |"
    sep = "|" + "-" * (width - 2) + "|"
    out = [top, ttl, sep]
    for ln in body_lines:
        out.append("| " + pad(ln, width - 4) + " |")
    out.append(top)
    return "\n".join(out)


# ----------------------------
# Progress animation
# ----------------------------

async def animate_progress(anchor: Message, chat_id: int, title: str, seconds: float = 2.0) -> Optional[Message]:
    """
    Sends a message and edits it with retro frames for `seconds`.
    If editing fails, falls back to sending new frames less often.
    """
    msg: Optional[Message] = None
    start = time.time()
    step = 0

    try:
        msg = await anchor.answer(html_pre(retro_frame(0, title, inner_w=34)), reply_markup=cancel_inline_kb(chat_id))
    except Exception:
        msg = None

    while time.time() - start < seconds:
        # cancel support
        task = RUNNING_TASK.get(chat_id)
        if task and task.cancelled():
            break

        frame = html_pre(retro_frame(step, title, inner_w=34))

        if msg is not None:
            try:
                await msg.edit_text(frame, reply_markup=cancel_inline_kb(chat_id))
            except Exception:
                # Telegram sometimes says "message can't be edited"
                msg = None
        else:
            # fallback: do not spam; send only occasionally
            if step % 4 == 0:
                try:
                    msg = await anchor.answer(frame, reply_markup=cancel_inline_kb(chat_id))
                except Exception:
                    msg = None

        step += 1
        await asyncio.sleep(0.25)

    return msg


# ----------------------------
# Review message composition
# ----------------------------

def build_verdict_text(chat_id: int, review: Dict[str, Any]) -> str:
    score = review.get("score", 0)
    verdict = (review.get("verdict") or "").strip()

    issues = review.get("issues") or []
    issues_lines: List[str] = []
    for i, it in enumerate(issues[:MAX_ISSUES], start=1):
        kind = "UX" if it.get("kind") == "ux" else "COPY"
        title = (it.get("title") or "").strip()
        problem = (it.get("problem") or "").strip()
        fix = (it.get("fix") or "").strip()

        # Keep it crisp (no tech details)
        row = f"{i}. [{kind}] {title}\n   What’s wrong: {problem}\n   Do this: {fix}"
        issues_lines.append(row)

    issues_block = "\n\n".join(issues_lines).strip()

    txt = (
        f"<b>{html_escape(t(chat_id, 'score'))}:</b> {html_escape(str(score))}/10\n\n"
        f"<b>{html_escape(t(chat_id, 'verdict'))}:</b>\n{html_escape(verdict)}"
    )
    if issues_block:
        txt += f"\n\n<b>{html_escape(t(chat_id, 'issues'))}:</b>\n{html_escape(issues_block)}"
    return txt


# ----------------------------
# Handlers
# ----------------------------

bot = Bot(BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()


@dp.callback_query(F.data == "cancel")
async def on_cancel(cb: CallbackQuery):
    chat_id = cb.message.chat.id if cb.message else cb.from_user.id
    task = RUNNING_TASK.get(chat_id)
    if task and not task.done():
        task.cancel()
    try:
        await cb.answer("OK")
    except Exception:
        pass
    # keep menu only at start and end of review, but cancel is an end-of-review event
    if cb.message:
        await cb.message.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))


@dp.message(Command("start"))
async def start(m: Message):
    chat_id = m.chat.id
    USER_LANG.setdefault(chat_id, "en")
    title = t(chat_id, "welcome_title")
    body = t(chat_id, "welcome_body")
    await m.answer(f"<b>{html_escape(title)}</b>\n\n{html_escape(body)}", reply_markup=main_menu_kb(chat_id))


@dp.message(Command("ping"))
async def ping(m: Message):
    # Kept for compatibility; menu now has channel button instead.
    await m.answer("pong", reply_markup=main_menu_kb(m.chat.id))


@dp.message(F.text)
async def on_text(m: Message):
    chat_id = m.chat.id
    text = (m.text or "").strip()

    # Language toggle button text
    if text in ("🌐 EN", "🌐 RU"):
        USER_LANG[chat_id] = "ru" if text == "🌐 RU" else "en"
        await m.answer("OK", reply_markup=main_menu_kb(chat_id))
        return

    # Menu buttons
    if text == TXT[lang_of(chat_id)]["menu_how"]:
        await m.answer(html_escape(t(chat_id, "how")), reply_markup=main_menu_kb(chat_id))
        return

    if text == TXT[lang_of(chat_id)]["menu_channel"]:
        await m.answer(html_escape(t(chat_id, "channel_msg")), reply_markup=main_menu_kb(chat_id))
        await m.answer(" ", reply_markup=channel_inline_kb())
        return

    if text == TXT[lang_of(chat_id)]["menu_drop"]:
        await m.answer(html_escape(t(chat_id, "need_input")), reply_markup=main_menu_kb(chat_id))
        return

    # Figma link?
    if is_probably_figma_link(text):
        url = extract_figma_url(text)
        if not url:
            await m.answer(html_escape(t(chat_id, "need_input")), reply_markup=main_menu_kb(chat_id))
            return
        await handle_review_input(m, figma_url=url)
        return

    # Otherwise: gentle nudge
    await m.answer(html_escape(t(chat_id, "need_input")), reply_markup=main_menu_kb(chat_id))


@dp.message(F.photo)
async def on_photo(m: Message):
    await handle_review_input(m, photo_message=m)


# ----------------------------
# Core pipeline
# ----------------------------

async def handle_review_input(m: Message, photo_message: Optional[Message] = None, figma_url: Optional[str] = None):
    chat_id = m.chat.id

    # Prevent parallel reviews per chat
    existing = RUNNING_TASK.get(chat_id)
    if existing and not existing.done():
        await m.answer(html_escape(t(chat_id, "busy")), reply_markup=main_menu_kb(chat_id))
        return

    task = asyncio.create_task(_review_task(m, photo_message=photo_message, figma_url=figma_url))
    RUNNING_TASK[chat_id] = task
    try:
        await task
    except asyncio.CancelledError:
        # already handled by cancel callback
        pass
    finally:
        # cleanup task
        if RUNNING_TASK.get(chat_id) is task:
            RUNNING_TASK.pop(chat_id, None)


async def _review_task(anchor: Message, photo_message: Optional[Message], figma_url: Optional[str]):
    chat_id = anchor.chat.id
    lang = lang_of(chat_id)

    if not BOT_TOKEN:
        await anchor.answer("BOT_TOKEN is not set.")
        return

    if not LLM_ENABLED:
        await anchor.answer(html_escape(t(chat_id, "llm_off")), reply_markup=main_menu_kb(chat_id))
        return

    if not OPENAI_API_KEY:
        await anchor.answer("OPENAI_API_KEY is not set.", reply_markup=main_menu_kb(chat_id))
        return

    # 1) Get image bytes
    img_bytes: Optional[bytes] = None

    await anchor.answer(html_escape(t(chat_id, "accepted")), reply_markup=None)

    if figma_url:
        await anchor.answer(html_escape(t(chat_id, "downloading_figma")), reply_markup=None)
        img_bytes = await asyncio.to_thread(download_figma_preview, figma_url)
        if not img_bytes:
            await anchor.answer(html_escape(t(chat_id, "figma_failed")), reply_markup=main_menu_kb(chat_id))
            return
        # Show the downloaded preview (so user sees what exactly got reviewed)
        try:
            await anchor.answer_photo(img_bytes, caption="Figma preview")
        except Exception:
            pass

    elif photo_message:
        # Download telegram photo
        try:
            file = await bot.get_file(photo_message.photo[-1].file_id)
            img_bytes = await bot.download_file(file.file_path)
            if hasattr(img_bytes, "read"):
                img_bytes = img_bytes.read()
        except Exception:
            await anchor.answer("Failed to download image.", reply_markup=main_menu_kb(chat_id))
            return

    else:
        await anchor.answer(html_escape(t(chat_id, "need_input")), reply_markup=main_menu_kb(chat_id))
        return

    # Cancel check
    task = RUNNING_TASK.get(chat_id)
    if task and task.cancelled():
        return

    # 2) Retro progress animation: analysis
    await animate_progress(anchor, chat_id, title=t(chat_id, "processing"), seconds=2.5)

    # 3) LLM review
    image_data_url = image_bytes_to_data_url(img_bytes)
    try:
        review = await asyncio.to_thread(llm_review, image_data_url, lang)
    except asyncio.CancelledError:
        return
    except Exception:
        await anchor.answer(html_escape(t(chat_id, "llm_bad")), reply_markup=main_menu_kb(chat_id))
        return

    # 4) Message #1: what I see
    what = (review.get("what_i_see") or "").strip()
    if not what:
        what = "(No description.)" if lang == "en" else "(Нет описания.)"
    msg1 = f"<b>{html_escape(t(chat_id, 'what_i_see'))}:</b>\n{html_escape(what)}"
    await anchor.answer(msg1, reply_markup=None)

    # 5) Message #2: verdict + recommendations + score + issues list
    msg2 = build_verdict_text(chat_id, review)
    await anchor.answer(msg2, reply_markup=None)

    # 6) Message #3: annotated image
    issues = review.get("issues") or []
    try:
        annotated = await asyncio.to_thread(annotate_image, img_bytes, issues)
        await anchor.answer_photo(annotated, caption="Annotations")
    except Exception:
        # If annotation fails, don’t crash the review
        pass

    # 7) Progress animation before ASCII concept (shorter)
    await animate_progress(anchor, chat_id, title=t(chat_id, "thinking"), seconds=1.6)

    # 8) Message #4: ASCII concept (monospace, no wrapping)
    concept = (review.get("ascii_concept") or "").strip()
    if not concept:
        concept = "ASCII concept not provided." if lang == "en" else "ASCII-концепт не получился."
    concept = hard_wrap_lines(concept, ASCII_CONCEPT_COLS)

    boxed = ascii_box(t(chat_id, "ascii_concept"), concept, width=ASCII_CONCEPT_COLS + 4)
    await anchor.answer(html_pre(boxed), reply_markup=None)

    # 9) End-of-review: show menu again (ONLY here)
    await anchor.answer(html_escape(t(chat_id, "done")), reply_markup=main_menu_kb(chat_id))


# ----------------------------
# Entrypoint
# ----------------------------

async def main():
    if not BOT_TOKEN:
        raise RuntimeError("Set BOT_TOKEN in environment")
    print(f"✅ Bot starting... LLM_ENABLED={LLM_ENABLED}, model={LLM_MODEL}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())