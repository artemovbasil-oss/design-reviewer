# bot.py
# Design Reviewer Telegram bot (Aiogram 3.7+)
# - Accepts screenshots (photos) and public Figma links (via oEmbed thumbnail)
# - Shows compact retro ASCII progress animation (loops until done)
# - Sends 4 outputs:
#   1) What I see
#   2) Verdict + recommendations
#   3) Annotated screenshot (boxes from LLM)
#   4) ASCII wireframe concept (monospace, width-limited to avoid mobile wrapping)
# - EN by default, RU toggle
# - Menu shown only after /start and at the end of each review
# - Cancel button works during processing

import asyncio
import base64
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
)
from aiogram.client.default import DefaultBotProperties

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in environment (Railway Variables or local export).")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ----------------------------
# State (in-memory)
# ----------------------------

LANG: Dict[int, str] = {}  # user_id -> "en" / "ru"
CANCEL_FLAGS: Dict[int, bool] = {}  # user_id -> True when cancelled
RUNNING_TASK: Dict[int, asyncio.Task] = {}  # user_id -> running processing task


# ----------------------------
# Helpers: i18n + HTML escaping
# ----------------------------

def lang_for(uid: int) -> str:
    return LANG.get(uid, "en")


def set_lang(uid: int, value: str) -> None:
    LANG[uid] = "ru" if value == "ru" else "en"


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


TXT = {
    "en": {
        "title": "Design Reviewer",
        "start": (
            "I'm your design review partner.\n\n"
            "Send me:\n"
            "• screenshots (images)\n"
            "• Figma frame links (public files)\n\n"
            "Tap “Submit for review” or just send a screenshot/link."
        ),
        "how": (
            "How it works:\n"
            "1) Send a screenshot or a public Figma frame link\n"
            "2) I analyze what’s on the screen\n"
            "3) I give a tough-but-fair review + a wireframe concept"
        ),
        "need_input": "Send a screenshot or a public Figma frame link.",
        "cancelled": "Cancelled.",
        "busy": "I’m already reviewing something. Hit Cancel, or wait for the result.",
        "no_llm": "LLM is disabled (no OPENAI_API_KEY). Add it to Railway Variables.",
        "llm_fail": "LLM error. Try again (or send a clearer / larger screenshot).",
        "not_image": "I need a screenshot image or a Figma link.",
        "figma_fetch_fail": "Couldn’t fetch a preview from that Figma link. Make sure the file is public.",
        "whatisee": "What I see",
        "verdict": "Verdict & recommendations",
        "annotated": "Annotated screenshot",
        "concept": "Wireframe concept (ASCII)",
        "score": "Score",
        "channel_msg": "Channel:",
        "open_channel": "Open @prodooktovy",
        "menu_submit": "Submit for review",
        "menu_how": "How it works?",
        "menu_lang": "Language: EN/RU",
        "menu_channel": "@prodooktovy",
        "btn_cancel": "Cancel",
        "progress_title": "Review in progress",
        "progress_wire": "Drafting wireframe",
        "preview": "Preview",
    },
    "ru": {
        "title": "Design Reviewer",
        "start": (
            "Я твой партнёр по дизайн-ревью.\n\n"
            "Принимаю:\n"
            "• скриншоты (картинки)\n"
            "• ссылки на фреймы Figma (если файл публичный)\n\n"
            "Жми «Закинуть на ревью» или просто отправь скрин/ссылку."
        ),
        "how": (
            "Как это работает:\n"
            "1) Отправь скриншот или публичную ссылку на Figma фрейм\n"
            "2) Я опишу, что вижу на экране\n"
            "3) Дам честный разбор + черновой вайрфрейм"
        ),
        "need_input": "Отправь скриншот или публичную ссылку на Figma фрейм.",
        "cancelled": "Отменено.",
        "busy": "Я уже делаю ревью. Нажми Cancel или дождись результата.",
        "no_llm": "LLM отключён (нет OPENAI_API_KEY). Добавь его в Railway Variables.",
        "llm_fail": "Ошибка LLM. Попробуй ещё раз (или пришли скрин крупнее/четче).",
        "not_image": "Мне нужен скриншот или ссылка на Figma.",
        "figma_fetch_fail": "Не смог скачать превью по ссылке Figma. Убедись, что файл публичный.",
        "whatisee": "Что я вижу",
        "verdict": "Вердикт и рекомендации",
        "annotated": "Аннотации на скрине",
        "concept": "Wireframe-концепт (ASCII)",
        "score": "Оценка",
        "channel_msg": "Канал:",
        "open_channel": "Открыть @prodooktovy",
        "menu_submit": "Закинуть на ревью",
        "menu_how": "Как это работает?",
        "menu_lang": "Язык: EN/RU",
        "menu_channel": "@prodooktovy",
        "btn_cancel": "Cancel",
        "progress_title": "Ревью в процессе",
        "progress_wire": "Собираю wireframe",
        "preview": "Превью",
    },
}


def t(uid: int, key: str) -> str:
    return TXT[lang_for(uid)].get(key, key)


# ----------------------------
# Keyboards
# ----------------------------

def main_menu_kb(uid: int) -> ReplyKeyboardMarkup:
    # 3 buttons total, last row contains channel + language
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=t(uid, "menu_submit"))],
            [KeyboardButton(text=t(uid, "menu_how"))],
            [KeyboardButton(text=t(uid, "menu_channel")), KeyboardButton(text=t(uid, "menu_lang"))],
        ],
        resize_keyboard=True,
        selective=True,
        input_field_placeholder=t(uid, "need_input"),
    )


def cancel_kb(uid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t(uid, "btn_cancel"), callback_data=f"cancel:{uid}")]
        ]
    )


def channel_inline_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Open @prodooktovy", url="https://t.me/prodooktovy")]
        ]
    )


# ----------------------------
# Progress animation (compact + looping)
# ----------------------------

def retro_bar_frame(step: int, width: int = 16) -> str:
    # compact retro "scanner" bar
    pos = step % width
    inside = ["·"] * width
    inside[pos] = "█"
    inside[(pos - 1) % width] = "▓"
    inside[(pos - 2) % width] = "▒"
    return (
        "┌" + "─" * width + "┐\n"
        "│" + "".join(inside) + "│\n"
        "└" + "─" * width + "┘"
    )


async def animate_progress_until_done(
    anchor: Message,
    uid: int,
    title: str,
    done_event: asyncio.Event,
    tick: float = 0.12,
) -> Message:
    """
    Sends ONE message and keeps editing it until done_event is set or cancelled.
    """
    msg = await anchor.answer(
        f"{escape_html(title)}\n<code>{escape_html(retro_bar_frame(0))}</code>",
        reply_markup=cancel_kb(uid),
    )

    step = 0
    while not done_event.is_set():
        if CANCEL_FLAGS.get(uid):
            break
        step += 1
        try:
            await msg.edit_text(
                f"{escape_html(title)}\n<code>{escape_html(retro_bar_frame(step))}</code>",
                reply_markup=cancel_kb(uid),
            )
        except Exception:
            # "message can't be edited" / throttling / etc — ignore, keep going
            pass
        await asyncio.sleep(tick)

    return msg


# ----------------------------
# Figma: fetch thumbnail via oEmbed (public links)
# ----------------------------

FIGMA_RE = re.compile(r"https?://www\.figma\.com/(file|design)/[A-Za-z0-9]+/[^?\s]+.*", re.I)

def is_figma_link(text: str) -> bool:
    return bool(FIGMA_RE.search(text or ""))


def http_get_bytes(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_figma_thumbnail(figma_url: str) -> Optional[bytes]:
    # cache-bust to avoid "same result for every link" in some proxy/CDN layers
    bust = str(int(time.time() * 1000))
    oembed = "https://www.figma.com/oembed?url=" + urllib.parse.quote(figma_url, safe="")
    oembed += ("&cb=" + bust)

    try:
        raw = http_get_bytes(oembed, timeout=20.0)
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        thumb = data.get("thumbnail_url")
        if not thumb:
            return None
        # additional cache-bust on thumbnail url
        sep = "&" if "?" in thumb else "?"
        thumb2 = thumb + f"{sep}cb={bust}"
        return http_get_bytes(thumb2, timeout=25.0)
    except Exception:
        return None


# ----------------------------
# Image: annotations + ascii width control
# ----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Pillow 10+: prefer textbbox
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return (box[2] - box[0], box[3] - box[1])
    except Exception:
        # fallback
        try:
            box = font.getbbox(text)
            return (box[2] - box[0], box[3] - box[1])
        except Exception:
            return (len(text) * 6, 10)


def draw_annotations(img_bytes: bytes, boxes: List[Dict[str, Any]]) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # default font (safe)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # monochrome look: white strokes on semi-transparent black fill
    for i, b in enumerate(boxes[:20], start=1):
        try:
            x1 = int(clamp01(float(b.get("x1", 0))) * W)
            y1 = int(clamp01(float(b.get("y1", 0))) * H)
            x2 = int(clamp01(float(b.get("x2", 0))) * W)
            y2 = int(clamp01(float(b.get("y2", 0))) * H)
            if x2 <= x1 or y2 <= y1:
                continue
            label = str(b.get("label") or f"#{i}")[:24]

            # translucent fill
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 255), width=3)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 30))

            # label
            if font:
                tw, th = _text_bbox(draw, label, font)
                pad = 4
                lx1, ly1 = x1, max(0, y1 - th - pad * 2)
                lx2, ly2 = x1 + tw + pad * 2, ly1 + th + pad * 2
                draw.rectangle([lx1, ly1, lx2, ly2], fill=(0, 0, 0, 200))
                draw.text((lx1 + pad, ly1 + pad), label, font=font, fill=(255, 255, 255, 255))
        except Exception:
            continue

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def compute_ascii_width_for_mobile(img_bytes: bytes) -> int:
    # conservative width to avoid wrapping in Telegram on phones
    # (monospace code block still wraps if too wide)
    # 32–38 is usually safe; pick 34 as a good default
    return 34


def enforce_ascii_lines(lines: List[str], width: int, height: int) -> List[str]:
    # truncate/pad each line to exact width and exact height
    fixed = []
    for ln in (lines or []):
        ln = ln.replace("\t", " ")
        if len(ln) > width:
            ln = ln[:width]
        fixed.append(ln.ljust(width))
        if len(fixed) >= height:
            break
    while len(fixed) < height:
        fixed.append(" " * width)
    return fixed


# ----------------------------
# OpenAI / LLM
# ----------------------------

def img_to_data_uri_png(img_bytes: bytes) -> str:
    # ensure PNG to keep things consistent
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


def parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # try to extract JSON object from text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start : end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        return None


def llm_request(prompt_text: str, data_uri: str) -> str:
    if not client:
        raise RuntimeError("No OpenAI client")

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": data_uri},
                ],
            }
        ],
    )

    # unify different sdk shapes
    try:
        return resp.output_text or ""
    except Exception:
        try:
            return resp.output[0].content[0].text  # type: ignore
        except Exception:
            return ""


async def llm_review_image(img_bytes: bytes, uid: int, ascii_w: int) -> Optional[Dict[str, Any]]:
    if not client:
        return {"error": "no_llm"}

    language = lang_for(uid)
    data_uri = img_to_data_uri_png(img_bytes)

    # Keep output strictly JSON to avoid Telegram HTML parsing crashes
    prompt = f"""
You are a tough-but-fair senior product designer doing a design review.

Return ONLY valid JSON. No markdown. No extra keys.

Language: "{language}" ("en" or "ru").
Be honest, no profanity.

Tasks:
1) Describe what you see on the screenshot (short, concrete).
2) Give verdict + recommendations (mix UI/UX + text, actionable).
   - Guess font family vibe only (e.g., "Inter-like / SF Pro-like / Roboto-like"), no exact sizes, no hex colors.
   - If something is good, praise it and say what exactly is good.
3) Provide a score from 1 to 10 (integer).
4) Provide up to 12 annotation boxes for major issues or key elements.
   - Coordinates are RELATIVE 0..1: x1,y1,x2,y2
   - label should be short (max 24 chars), in the chosen language
5) Provide an ASCII wireframe concept that fits exactly within width={ascii_w} characters per line.
   - Provide as an array of strings (ascii_concept).
   - Each line MUST be <= {ascii_w} chars.
   - Use simple box-drawing / ASCII, make it feel like a wireframe.

JSON schema:
{{
  "language": "{language}",
  "score_10": 7,
  "what_i_see": "...",
  "verdict": "...",
  "boxes": [{{"x1":0.1,"y1":0.2,"x2":0.4,"y2":0.3,"label":"..."}}],
  "ascii_width": {ascii_w},
  "ascii_height": 18,
  "ascii_concept": ["..."]
}}
""".strip()

    # run sync openai call in a thread so Cancel can interrupt
    try:
        text = await asyncio.to_thread(llm_request, prompt, data_uri)
        data = parse_json_safe(text)
        if not data:
            return None
        return data
    except Exception:
        return None


# ----------------------------
# Review pipeline
# ----------------------------

async def send_channel(uid: int, m: Message) -> None:
    lang = lang_for(uid)
    text = f"{t(uid,'channel_msg')} @prodooktovy"
    await m.answer(escape_html(text), reply_markup=channel_inline_kb())


async def do_review(message: Message, bot: Bot, img_bytes: bytes, uid: int) -> None:
    # mark running
    CANCEL_FLAGS[uid] = False

    # 1) progress spinner while LLM runs
    done = asyncio.Event()
    spinner_task = asyncio.create_task(
        animate_progress_until_done(
            message,
            uid,
            title=t(uid, "progress_title"),
            done_event=done,
            tick=0.12,
        )
    )

    try:
        if not client:
            done.set()
            await message.answer(escape_html(t(uid, "no_llm")), reply_markup=main_menu_kb(uid))
            return

        ascii_w = compute_ascii_width_for_mobile(img_bytes)

        result = await llm_review_image(img_bytes, uid, ascii_w)

        # stop spinner
        done.set()
        try:
            await spinner_task
        except Exception:
            pass

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            return

        if not result:
            await message.answer(escape_html(t(uid, "llm_fail")), reply_markup=main_menu_kb(uid))
            return

        if result.get("error") == "no_llm":
            await message.answer(escape_html(t(uid, "no_llm")), reply_markup=main_menu_kb(uid))
            return

        # enforce language in text output
        result["language"] = lang_for(uid)

        score = result.get("score_10", None)
        try:
            score_int = int(score)
        except Exception:
            score_int = 0
        score_int = max(1, min(10, score_int))

        # 1) What I see
        msg1 = (
            f"<b>{escape_html(t(uid,'whatisee'))}</b>\n"
            f"{escape_html(str(result.get('what_i_see','')).strip())}\n\n"
            f"<b>{escape_html(t(uid,'score'))}: {score_int}/10</b>"
        )
        await message.answer(msg1)

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            return

        # 2) Verdict + recommendations
        msg2 = f"<b>{escape_html(t(uid,'verdict'))}</b>\n{escape_html(str(result.get('verdict','')).strip())}"
        await message.answer(msg2)

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            return

        # 3) Annotated screenshot
        boxes = result.get("boxes") or []
        if isinstance(boxes, list) and boxes:
            try:
                annotated_bytes = draw_annotations(img_bytes, boxes)
                cap = f"{t(uid,'annotated')}"
                photo = BufferedInputFile(annotated_bytes, filename="annotated.png")
                await message.answer_photo(photo=photo, caption=escape_html(cap))
            except Exception:
                # if annotation fails, do not break the whole flow
                pass

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            return

        # 4) ASCII wireframe concept
        # small second spinner for wireframe (quick)
        done2 = asyncio.Event()
        spinner2 = asyncio.create_task(
            animate_progress_until_done(
                message,
                uid,
                title=t(uid, "progress_wire"),
                done_event=done2,
                tick=0.12,
            )
        )

        try:
            width = int(result.get("ascii_width") or ascii_w)
            height = int(result.get("ascii_height") or 18)
            width = max(26, min(42, width))   # keep it phone-safe
            height = max(12, min(26, height)) # not too tall
            lines = result.get("ascii_concept") or []
            if not isinstance(lines, list):
                lines = []
            fixed = enforce_ascii_lines([str(x) for x in lines], width=width, height=height)
            concept_block = "\n".join(fixed)
        finally:
            done2.set()
            try:
                await spinner2
            except Exception:
                pass

        msg4 = f"<b>{escape_html(t(uid,'concept'))}</b>\n<code>{escape_html(concept_block)}</code>"
        await message.answer(msg4)

        # menu at the end
        await message.answer(escape_html(t(uid, "need_input")), reply_markup=main_menu_kb(uid))

    finally:
        done.set()
        if not spinner_task.done():
            spinner_task.cancel()


# ----------------------------
# Telegram handlers
# ----------------------------

router = Router()


@router.callback_query(F.data.startswith("cancel:"))
async def on_cancel(cb: CallbackQuery):
    try:
        uid = cb.from_user.id
    except Exception:
        return

    CANCEL_FLAGS[uid] = True

    # cancel running task if exists
    task = RUNNING_TASK.get(uid)
    if task and not task.done():
        task.cancel()

    try:
        await cb.answer("OK", show_alert=False)
    except Exception:
        pass

    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass


@router.message(CommandStart())
async def on_start(m: Message):
    uid = m.from_user.id
    if uid not in LANG:
        set_lang(uid, "en")

    await m.answer(
        escape_html(t(uid, "start")),
        reply_markup=main_menu_kb(uid),
    )


@router.message(F.text)
async def on_text(m: Message, bot: Bot):
    uid = m.from_user.id
    txt = (m.text or "").strip()

    # menu actions
    if txt == t(uid, "menu_how"):
        await m.answer(escape_html(t(uid, "how")), reply_markup=main_menu_kb(uid))
        return

    if txt == t(uid, "menu_lang"):
        # toggle
        new_lang = "ru" if lang_for(uid) == "en" else "en"
        set_lang(uid, new_lang)
        await m.answer(escape_html(t(uid, "start")), reply_markup=main_menu_kb(uid))
        return

    if txt == t(uid, "menu_channel"):
        await send_channel(uid, m)
        return

    if txt == t(uid, "menu_submit"):
        await m.answer(escape_html(t(uid, "need_input")))
        return

    # figma link
    if is_figma_link(txt):
        if RUNNING_TASK.get(uid) and not RUNNING_TASK[uid].done():
            await m.answer(escape_html(t(uid, "busy")))
            return

        thumb = fetch_figma_thumbnail(txt)
        if not thumb:
            await m.answer(escape_html(t(uid, "figma_fetch_fail")), reply_markup=main_menu_kb(uid))
            return

        # show preview image first (optional)
        try:
            photo = BufferedInputFile(thumb, filename="figma_preview.png")
            await m.answer_photo(photo=photo, caption=escape_html(t(uid, "preview")))
        except Exception:
            pass

        async def _run():
            await do_review(m, bot, thumb, uid)

        task = asyncio.create_task(_run())
        RUNNING_TASK[uid] = task
        return

    # otherwise
    await m.answer(escape_html(t(uid, "not_image")), reply_markup=main_menu_kb(uid))


@router.message(F.photo)
async def on_photo(m: Message, bot: Bot):
    uid = m.from_user.id

    if RUNNING_TASK.get(uid) and not RUNNING_TASK[uid].done():
        await m.answer(escape_html(t(uid, "busy")))
        return

    # download best size
    photo = m.photo[-1]
    file = await bot.get_file(photo.file_id)
    img_bytes = await bot.download_file(file.file_path)
    data = img_bytes.read() if hasattr(img_bytes, "read") else bytes(img_bytes)

    async def _run():
        await do_review(m, bot, data, uid)

    task = asyncio.create_task(_run())
    RUNNING_TASK[uid] = task


# ----------------------------
# App entry
# ----------------------------

async def main():
    dp = Dispatcher()
    dp.include_router(router)

    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())