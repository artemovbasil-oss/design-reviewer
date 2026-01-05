import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# =========================
# Config
# =========================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# model for vision
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN env var")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

router = Router()

# =========================
# State (in-memory)
# =========================
USER_LANG: Dict[int, str] = {}  # "en" | "ru"
RUNNING_TASKS: Dict[int, asyncio.Task] = {}
CANCEL_FLAGS: Dict[int, bool] = {}

# =========================
# UI strings
# =========================
STR = {
    "en": {
        "hello_title": "Design Reviewer (retro edition)",
        "hello_body": (
            "I’m your picky design buddy.\n\n"
            "Send me:\n"
            "• a UI screenshot\n"
            "• a Figma frame link (public file)\n\n"
            "I’ll reply with:\n"
            "1) what I see\n"
            "2) verdict + recommendations\n"
            "3) annotated screenshot\n"
            "4) an ASCII concept prototype\n"
        ),
        "btn_review": "Drop for review",
        "btn_how": "How it works?",
        "btn_lang": "EN/RU",
        "btn_cancel": "Cancel",
        "btn_channel": "@prodooktovy",
        "how": (
            "How it works:\n"
            "1) Send a screenshot or a public Figma frame link\n"
            "2) Watch the retro progress\n"
            "3) Get a blunt review + annotated screenshot + ASCII concept"
        ),
        "need_input": "Send me a screenshot or a public Figma frame link.",
        "cancelled": "Cancelled. Send another screenshot/link when ready.",
        "busy": "I’m already reviewing something. Hit Cancel if you want to stop it.",
        "llm_fail": "LLM error. Try again (or send a clearer / larger screenshot).",
        "score": "Score",
        "whatisee": "What I see",
        "verdict": "Verdict & recommendations",
        "annotated": "Annotated",
        "concept": "ASCII concept prototype",
    },
    "ru": {
        "hello_title": "Design Reviewer (retro edition)",
        "hello_body": (
            "Я твой придирчивый дизайн-товарищ.\n\n"
            "Кидай сюда:\n"
            "• скрин интерфейса\n"
            "• ссылку на фрейм Figma (если файл публичный)\n\n"
            "Я отвечу:\n"
            "1) что вижу\n"
            "2) вердикт + рекомендации\n"
            "3) аннотированный скрин\n"
            "4) ASCII-концепт прототип"
        ),
        "btn_review": "Закинуть на ревью",
        "btn_how": "Как это работает?",
        "btn_lang": "EN/RU",
        "btn_cancel": "Отмена",
        "btn_channel": "@prodooktovy",
        "how": (
            "Как это работает:\n"
            "1) Отправь скрин или ссылку на публичный фрейм Figma\n"
            "2) Посмотри ретро-прогресс\n"
            "3) Получи жесткое ревью + аннотированный скрин + ASCII-концепт"
        ),
        "need_input": "Пришли скриншот или ссылку на публичный фрейм Figma.",
        "cancelled": "Отменено. Пришли следующий скрин/ссылку, когда будешь готов.",
        "busy": "Я уже в процессе ревью. Нажми Отмена, если хочешь остановить.",
        "llm_fail": "Ошибка LLM. Попробуй ещё раз (или пришли скрин крупнее/четче).",
        "score": "Оценка",
        "whatisee": "Что я вижу",
        "verdict": "Вердикт и рекомендации",
        "annotated": "Аннотации",
        "concept": "ASCII-концепт прототип",
    },
}

# =========================
# Helpers
# =========================
def lang_for(user_id: int) -> str:
    return USER_LANG.get(user_id, "en")


def t(user_id: int, key: str) -> str:
    return STR[lang_for(user_id)][key]


def escape_html(s: str) -> str:
    # minimal HTML escape
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main_menu_kb(user_id: int) -> InlineKeyboardMarkup:
    L = lang_for(user_id)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=STR[L]["btn_review"], callback_data="review")],
            [InlineKeyboardButton(text=STR[L]["btn_how"], callback_data="how")],
            [
                InlineKeyboardButton(text=STR[L]["btn_channel"], url="https://t.me/prodooktovy"),
                InlineKeyboardButton(text=STR[L]["btn_lang"], callback_data="toggle_lang"),
            ],
        ]
    )


def cancel_kb(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=t(user_id, "btn_cancel"), callback_data="cancel")]]
    )


FIGMA_URL_RE = re.compile(r"https?://(www\.)?figma\.com/(file|design)/[^\s]+", re.IGNORECASE)


def extract_figma_url(text: str) -> Optional[str]:
    m = FIGMA_URL_RE.search(text or "")
    return m.group(0) if m else None


def is_image_message(m: Message) -> bool:
    return bool(m.photo) or (m.document and (m.document.mime_type or "").startswith("image/"))


async def download_tg_image_bytes(m: Message, bot: Bot) -> Optional[bytes]:
    if m.photo:
        file_id = m.photo[-1].file_id
    elif m.document and (m.document.mime_type or "").startswith("image/"):
        file_id = m.document.file_id
    else:
        return None
    file = await bot.get_file(file_id)
    bio = BytesIO()
    await bot.download_file(file.file_path, destination=bio)
    return bio.getvalue()


async def fetch_url_bytes(url: str, timeout: float = 20.0) -> bytes:
    # aiogram installs aiohttp; use it to avoid extra deps.
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.read()


async def figma_preview_image(figma_url: str) -> Optional[bytes]:
    """
    Uses Figma oEmbed (no token) to get thumbnail_url for public files.
    Avoids caching issues by adding cache-buster.
    """
    try:
        import aiohttp
        oembed = f"https://www.figma.com/oembed?url={figma_url}"
        # cache-buster
        oembed += f"&cb={int(time.time()*1000)}"

        async with aiohttp.ClientSession() as session:
            async with session.get(oembed, timeout=20) as r:
                if r.status != 200:
                    return None
                data = await r.json(content_type=None)
                thumb = data.get("thumbnail_url") or data.get("thumbnailUrl")
                if not thumb:
                    return None
            # fetch thumbnail bytes
            async with session.get(thumb + f"&cb={int(time.time()*1000)}", timeout=25) as r2:
                if r2.status != 200:
                    return None
                return await r2.read()
    except Exception:
        return None


# =========================
# Retro ASCII progress (compact + fast)
# =========================
def retro_bar_frame(step: int, width: int = 22) -> str:
    # a tiny, phone-friendly bar
    # width includes only inside area
    pos = step % width
    inside = ["·"] * width
    inside[pos] = "█"
    return "┌" + "─" * width + "┐\n" + "│" + "".join(inside) + "│\n" + "└" + "─" * width + "┘"


async def animate_progress(anchor: Message, user_id: int, title: str, seconds: float = 2.2) -> Message:
    """
    Sends ONE message, then edits it quickly.
    If edit fails (message can't be edited), falls back to sending a new one.
    """
    start = time.time()
    msg = await anchor.answer(
        f"{escape_html(title)}\n<code>{escape_html(retro_bar_frame(0))}</code>",
        reply_markup=cancel_kb(user_id),
    )
    step = 0
    while time.time() - start < seconds:
        if CANCEL_FLAGS.get(user_id):
            break
        step += 1
        try:
            await msg.edit_text(
                f"{escape_html(title)}\n<code>{escape_html(retro_bar_frame(step))}</code>",
                reply_markup=cancel_kb(user_id),
            )
        except Exception:
            # Can't edit (or race). Create a new message and continue editing that one.
            msg = await anchor.answer(
                f"{escape_html(title)}\n<code>{escape_html(retro_bar_frame(step))}</code>",
                reply_markup=cancel_kb(user_id),
            )
        await asyncio.sleep(0.12)
    return msg


# =========================
# LLM tool schema (strict JSON via function call)
# =========================
REVIEW_TOOL = [
    {
        "type": "function",
        "name": "deliver_review",
        "description": "Return a structured design review for the provided UI screenshot.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "language": {"type": "string", "enum": ["en", "ru"]},
                "score_10": {"type": "integer", "minimum": 1, "maximum": 10},
                "what_i_see": {"type": "string"},
                "verdict": {"type": "string"},
                "boxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "x1": {"type": "number", "minimum": 0, "maximum": 1},
                            "y1": {"type": "number", "minimum": 0, "maximum": 1},
                            "x2": {"type": "number", "minimum": 0, "maximum": 1},
                            "y2": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "x1", "y1", "x2", "y2"],
                    },
                },
                "ascii_concept": {
                    "type": "array",
                    "description": "Monospace ASCII prototype lines. MUST fit within max_width characters per line.",
                    "items": {"type": "string"},
                },
                "ascii_width": {"type": "integer", "minimum": 20, "maximum": 44},
            },
            "required": ["language", "score_10", "what_i_see", "verdict", "boxes", "ascii_concept", "ascii_width"],
        },
        "strict": True,
    }
]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def normalize_boxes(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for b in raw or []:
        x1, y1, x2, y2 = clamp01(b["x1"]), clamp01(b["y1"]), clamp01(b["x2"]), clamp01(b["y2"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # drop tiny junk boxes
        if (x2 - x1) < 0.02 or (y2 - y1) < 0.02:
            continue
        out.append({"label": str(b["label"])[:28], "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out[:14]


def enforce_width(lines: List[str], maxw: int) -> List[str]:
    cleaned = []
    for ln in (lines or []):
        ln = ln.rstrip("\n\r")
        if len(ln) > maxw:
            ln = ln[:maxw]
        cleaned.append(ln)
    # Ensure at least something
    return cleaned[:28] if cleaned else ["(no concept)"]


def llm_review_image(img_bytes: bytes, user_id: int) -> Optional[Dict[str, Any]]:
    L = lang_for(user_id)
    maxw = 38  # good for phone; keep safe (prevents wrap)
    tone = {
        "en": (
            "You are a senior product designer doing a blunt, helpful design review. "
            "No profanity. Be honest, picky, but fair. Praise what is good. "
            "Do NOT output any technical color codes or pixel measurements. "
            "You may GUESS font family style only (e.g., 'looks like Inter / SF / Roboto-ish'), no exact sizes. "
            "Focus on hierarchy, spacing, alignment, clarity, consistency, UX flows, and text quality."
        ),
        "ru": (
            "Ты — сеньор продакт-дизайнер и делаешь жесткое, но полезное ревью. "
            "Без мата. Честно придирайся, но по делу. Хорошее — тоже отмечай. "
            "НЕ давай технических кодов цветов или пиксельных размеров. "
            "Про шрифт — только предположение семейства (Inter/SF/Roboto-ish), без размеров. "
            "Фокус: иерархия, сетка, отступы, выравнивание, консистентность, UX, и качество текста."
        ),
    }[L]

    instructions = (
        f"{tone}\n\n"
        f"Return the result ONLY by calling the tool deliver_review.\n"
        f"language must be '{L}'.\n"
        f"ascii_width must be <= {maxw}.\n"
        f"ascii_concept: make a meaningful, more detailed ASCII wireframe concept for the SAME screen.\n"
        f"Every ascii_concept line MUST be <= ascii_width characters.\n"
        f"boxes: pick up to ~10 UI areas that your recommendations refer to.\n"
        f"what_i_see + verdict must be in the chosen language.\n"
    )

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            tools=REVIEW_TOOL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Review this UI screenshot."},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    ],
                }
            ],
            instructions=instructions,
        )

        # Find the function call
        for item in resp.output:
            if getattr(item, "type", None) == "function_call" and getattr(item, "name", "") == "deliver_review":
                args = json.loads(item.arguments)
                args["boxes"] = normalize_boxes(args.get("boxes", []))
                args["ascii_width"] = min(int(args.get("ascii_width", maxw)), maxw)
                args["ascii_concept"] = enforce_width(args.get("ascii_concept", []), args["ascii_width"])
                return args

        return None
    except Exception:
        return None


# =========================
# Annotation drawing
# =========================
def pick_font(size: int = 14) -> ImageFont.FreeTypeFont:
    # try common fonts; fallback to default
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Pillow 10+: use textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_annotations(image_bytes: bytes, boxes: List[Dict[str, Any]]) -> bytes:
    im = Image.open(BytesIO(image_bytes)).convert("RGBA")
    w, h = im.size
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    font = pick_font(14)

    # monochrome style: white boxes with black shadow
    for i, b in enumerate(boxes, start=1):
        x1 = int(b["x1"] * w)
        y1 = int(b["y1"] * h)
        x2 = int(b["x2"] * w)
        y2 = int(b["y2"] * h)
        label = f"{i}. {b['label']}"
        # shadow rect
        d.rectangle([x1 + 2, y1 + 2, x2 + 2, y2 + 2], outline=(0, 0, 0, 180), width=4)
        # main rect
        d.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 240), width=3)

        tw, th = text_size(d, label, font)
        pad = 6
        bx1, by1 = x1, max(0, y1 - (th + pad * 2))
        bx2, by2 = x1 + tw + pad * 2, y1
        # label bg
        d.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, 200))
        d.text((bx1 + pad, by1 + pad), label, font=font, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(im, overlay).convert("RGB")
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# =========================
# Review pipeline
# =========================
async def do_review(message: Message, bot: Bot, img_bytes: bytes, user_id: int):
    CANCEL_FLAGS[user_id] = False

    # progress 1
    await animate_progress(message, user_id, title=("Review in progress" if lang_for(user_id) == "en" else "Ревью в процессе"))

    if CANCEL_FLAGS.get(user_id):
        await message.answer(escape_html(t(user_id, "cancelled")), reply_markup=main_menu_kb(user_id))
        return

    result = llm_review_image(img_bytes, user_id)
    if not result:
        await message.answer(escape_html(t(user_id, "llm_fail")), reply_markup=main_menu_kb(user_id))
        return

    # Ensure language respected (hard override)
    if result.get("language") != lang_for(user_id):
        result["language"] = lang_for(user_id)

    # 1) What I see
    score_line = f"{t(user_id,'score')}: {int(result['score_10'])}/10"
    what_text = f"<b>{escape_html(t(user_id,'whatisee'))}</b>\n{escape_html(result['what_i_see'])}\n\n<b>{escape_html(score_line)}</b>"
    await message.answer(what_text)

    if CANCEL_FLAGS.get(user_id):
        await message.answer(escape_html(t(user_id, "cancelled")), reply_markup=main_menu_kb(user_id))
        return

    # 2) Verdict
    verdict_text = f"<b>{escape_html(t(user_id,'verdict'))}</b>\n{escape_html(result['verdict'])}"
    await message.answer(verdict_text)

    if CANCEL_FLAGS.get(user_id):
        await message.answer(escape_html(t(user_id, "cancelled")), reply_markup=main_menu_kb(user_id))
        return

    # 3) Annotated screenshot
    boxes = result.get("boxes", [])
    if boxes:
        try:
            annotated_bytes = draw_annotations(img_bytes, boxes)
            cap = f"{t(user_id,'annotated')}: {len(boxes)}"
            photo = BufferedInputFile(annotated_bytes, filename="annotated.png")
            await message.answer_photo(photo=photo, caption=escape_html(cap), reply_markup=None)
        except Exception:
            # don't crash the whole review
            pass

    if CANCEL_FLAGS.get(user_id):
        await message.answer(escape_html(t(user_id, "cancelled")), reply_markup=main_menu_kb(user_id))
        return

    # progress 2 (between annotated and concept)
    await animate_progress(message, user_id, title=("Drafting concept" if lang_for(user_id) == "en" else "Собираю концепт"), seconds=1.8)

    # 4) ASCII concept prototype (monospace, no wrap)
    width = int(result.get("ascii_width", 38))
    lines = enforce_width(result.get("ascii_concept", []), width)

    concept_block = "\n".join(lines)
    concept_msg = f"<b>{escape_html(t(user_id,'concept'))}</b>\n<code>{escape_html(concept_block)}</code>"
    await message.answer(concept_msg)

    # menu at the end
    await message.answer(escape_html(t(user_id, "need_input")), reply_markup=main_menu_kb(user_id))


async def start_review_from_image(message: Message, bot: Bot, img_bytes: bytes):
    user_id = message.from_user.id
    if user_id in RUNNING_TASKS and not RUNNING_TASKS[user_id].done():
        await message.answer(escape_html(t(user_id, "busy")), reply_markup=None)
        return

    task = asyncio.create_task(do_review(message, bot, img_bytes, user_id))
    RUNNING_TASKS[user_id] = task
    try:
        await task
    finally:
        RUNNING_TASKS.pop(user_id, None)
        CANCEL_FLAGS[user_id] = False


async def start_review_from_figma_link(message: Message, bot: Bot, figma_url: str):
    user_id = message.from_user.id
    if user_id in RUNNING_TASKS and not RUNNING_TASKS[user_id].done():
        await message.answer(escape_html(t(user_id, "busy")), reply_markup=None)
        return

    # show preview if we can
    await animate_progress(message, user_id, title=("Fetching Figma preview" if lang_for(user_id) == "en" else "Тащу превью из Figma"), seconds=1.6)
    img = await figma_preview_image(figma_url)
    if not img:
        await message.answer(
            escape_html(
                ("Couldn't fetch Figma preview. Make sure the file is public." if lang_for(user_id) == "en"
                 else "Не смог скачать превью. Проверь, что файл публичный.")
            ),
            reply_markup=main_menu_kb(user_id),
        )
        return

    # send preview image first (nice UX)
    try:
        photo = BufferedInputFile(img, filename="figma_preview.png")
        await message.answer_photo(photo=photo, caption=escape_html("Figma preview" if lang_for(user_id) == "en" else "Превью Figma"))
    except Exception:
        pass

    await start_review_from_image(message, bot, img)


# =========================
# Handlers
# =========================
@router.message(CommandStart())
async def on_start(m: Message):
    user_id = m.from_user.id
    if user_id not in USER_LANG:
        USER_LANG[user_id] = "en"

    title = STR[lang_for(user_id)]["hello_title"]
    body = STR[lang_for(user_id)]["hello_body"]
    await m.answer(f"<b>{escape_html(title)}</b>\n\n{escape_html(body)}", reply_markup=main_menu_kb(user_id))


@router.callback_query(F.data == "how")
async def on_how(cb: CallbackQuery):
    user_id = cb.from_user.id
    await cb.answer()
    await cb.message.answer(escape_html(STR[lang_for(user_id)]["how"]), reply_markup=main_menu_kb(user_id))


@router.callback_query(F.data == "review")
async def on_review_btn(cb: CallbackQuery):
    user_id = cb.from_user.id
    await cb.answer()
    await cb.message.answer(escape_html(t(user_id, "need_input")), reply_markup=None)


@router.callback_query(F.data == "toggle_lang")
async def on_toggle_lang(cb: CallbackQuery):
    user_id = cb.from_user.id
    USER_LANG[user_id] = "ru" if lang_for(user_id) == "en" else "en"
    await cb.answer("OK")
    # Update menu in place if possible
    try:
        await cb.message.edit_reply_markup(reply_markup=main_menu_kb(user_id))
    except Exception:
        await cb.message.answer("OK", reply_markup=main_menu_kb(user_id))


@router.callback_query(F.data == "cancel")
async def on_cancel(cb: CallbackQuery):
    user_id = cb.from_user.id
    CANCEL_FLAGS[user_id] = True
    await cb.answer("Cancelled" if lang_for(user_id) == "en" else "Отменено")
    await cb.message.answer(escape_html(t(user_id, "cancelled")), reply_markup=main_menu_kb(user_id))


@router.message(F.text)
async def on_text(m: Message, bot: Bot):
    user_id = m.from_user.id
    figma_url = extract_figma_url(m.text or "")
    if figma_url:
        await start_review_from_figma_link(m, bot, figma_url)
        return

    # otherwise: show short help (no menu spam)
    await m.answer(escape_html(t(user_id, "need_input")), reply_markup=None)


@router.message(F.photo | F.document)
async def on_image(m: Message, bot: Bot):
    user_id = m.from_user.id
    if not is_image_message(m):
        await m.answer(escape_html(t(user_id, "need_input")), reply_markup=None)
        return

    img_bytes = await download_tg_image_bytes(m, bot)
    if not img_bytes:
        await m.answer(escape_html(t(user_id, "llm_fail")), reply_markup=main_menu_kb(user_id))
        return
    await start_review_from_image(m, bot, img_bytes)


# =========================
# Main
# =========================
async def main():
    dp = Dispatcher()
    dp.include_router(router)

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())