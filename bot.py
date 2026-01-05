# bot.py
# Bereke Design Reviewer Bot (EN default, RU toggle)
# - Accepts screenshots (photos) and public Figma frame links
# - Shows compact retro ASCII progress (single message, edited)
# - Sends:
#   1) What I see
#   2) Verdict + recommendations (UX + copy), score /10
#   3) Annotated screenshot preview
#   4) ASCII concept (monospace), with another short ASCII progress
# - Menu (reply keyboard) appears only after /start and after each review
# - Inline Cancel button during processing

import asyncio
import base64
import io
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.exceptions import TelegramBadRequest

from PIL import Image, ImageDraw, ImageFont
import pytesseract

# Optional OpenAI (if installed + key provided)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# -------------------------
# Config (ENV only, no dotenv)
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
OCR_LANG = os.getenv("OCR_LANG", "rus+eng").strip()

# ASCII concept safe width for phones
ASCII_CONCEPT_COLS = 32

# Product design channel
CHANNEL_HANDLE = "@prodooktovy"
CHANNEL_URL = "https://t.me/prodooktovy"

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in environment variables")

# -------------------------
# State
# -------------------------
LANG: Dict[int, str] = {}  # chat_id -> "EN" or "RU"
RUNNING_TASK: Dict[int, asyncio.Task] = {}  # chat_id -> task


# -------------------------
# i18n
# -------------------------
TEXT = {
    "EN": {
        "brand": "Design Reviewer",
        "start": (
            "<b>Design Reviewer</b>\n"
            "Send me a screenshot or a public Figma frame link.\n"
            "I’ll nitpick your UI/UX and copy like a senior teammate.\n\n"
            "Use the menu below or just send a screenshot/link."
        ),
        "menu_review": "Send for review",
        "menu_how": "How it works?",
        "menu_lang_en": "Language: EN",
        "menu_lang_ru": "Language: RU",
        "how": (
            "<b>How it works</b>\n"
            "1) Send a screenshot (image) or a public Figma frame link\n"
            "2) I review what I see + UX + text\n"
            "3) I send annotations + an ASCII concept"
        ),
        "bad_link": "I can’t parse that link. Send a screenshot or a Figma link with node-id.",
        "figma_private": "Looks like this Figma file isn’t public. Make it public (view access) and try again.",
        "cancelled": "Cancelled. Send another screenshot/link when ready.",
        "processing": "Review in progress",
        "processing2": "Concept in progress",
        "what_i_see_title": "<b>1) What I see</b>",
        "verdict_title": "<b>2) Verdict + recommendations</b>",
        "anno_title": "<b>3) Annotations</b>",
        "concept_title": "<b>4) ASCII concept</b>",
        "llm_fail": "LLM error. Try again (or send a clearer / larger screenshot).",
        "no_llm": "LLM is disabled. I can only do OCR-based notes right now.",
        "cancel_btn": "Cancel",
        "score": "Score",
        "lang_switched_en": "Language switched to EN.",
        "lang_switched_ru": "Язык переключен на RU.",
    },
    "RU": {
        "brand": "Дизайн-ревьюер",
        "start": (
            "<b>Дизайн-ревьюер</b>\n"
            "Кидай скриншот или публичную ссылку на Figma-фрейм.\n"
            "Я докопаюсь до UI/UX и текста как старший товарищ.\n\n"
            "Жми кнопки снизу или просто отправь скрин/ссылку."
        ),
        "menu_review": "Закинуть на ревью",
        "menu_how": "Как это работает?",
        "menu_lang_en": "Язык: EN",
        "menu_lang_ru": "Язык: RU",
        "how": (
            "<b>Как это работает</b>\n"
            "1) Отправь скриншот или публичную ссылку на фрейм Figma\n"
            "2) Я разберу, что вижу + UX + тексты\n"
            "3) Пришлю аннотации + ASCII-концепт"
        ),
        "bad_link": "Ссылку не понял. Пришли скриншот или Figma-ссылку с node-id.",
        "figma_private": "Похоже, файл Figma не публичный. Открой доступ на просмотр и попробуй снова.",
        "cancelled": "Отменил. Готов — кидай следующий скрин/ссылку.",
        "processing": "Ревью в процессе",
        "processing2": "Концепт в процессе",
        "what_i_see_title": "<b>1) Что я вижу</b>",
        "verdict_title": "<b>2) Вердикт и рекомендации</b>",
        "anno_title": "<b>3) Аннотации</b>",
        "concept_title": "<b>4) ASCII-концепт</b>",
        "llm_fail": "Ошибка LLM. Попробуй ещё раз (или пришли скрин крупнее/четче).",
        "no_llm": "LLM выключен. Сейчас могу только OCR-заметки.",
        "cancel_btn": "Отмена",
        "score": "Оценка",
        "lang_switched_en": "Language switched to EN.",
        "lang_switched_ru": "Язык переключен на RU.",
    },
}


def lang_of(chat_id: int) -> str:
    return LANG.get(chat_id, "EN")


def t(chat_id: int, key: str) -> str:
    return TEXT[lang_of(chat_id)][key]


# -------------------------
# Keyboards (reply keyboard only at start + end of review)
# -------------------------
def main_menu_kb(chat_id: int) -> ReplyKeyboardMarkup:
    L = lang_of(chat_id)
    if L == "EN":
        lang_btn = KeyboardButton(text=TEXT["EN"]["menu_lang_ru"])  # switch to RU
    else:
        lang_btn = KeyboardButton(text=TEXT["RU"]["menu_lang_en"])  # switch to EN

    # 3 rows:
    # 1) Review
    # 2) How it works?
    # 3) Channel + Lang in one row (as requested)
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=t(chat_id, "menu_review"))],
            [KeyboardButton(text=t(chat_id, "menu_how"))],
            [
                KeyboardButton(text="Product design channel" if L == "EN" else "Канал о продуктовом дизайне"),
                lang_btn,
            ],
        ],
        resize_keyboard=True,
        input_field_placeholder=("Send screenshot or Figma link" if L == "EN" else "Скриншот или ссылка на Figma"),
        selective=False,
    )


def cancel_inline_kb(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=t(chat_id, "cancel_btn"), callback_data="cancel_review")]]
    )


# -------------------------
# HTML safety
# -------------------------
def html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def html_pre(s: str) -> str:
    # monospace block
    return f"<pre>{html_escape(s)}</pre>"


# -------------------------
# Retro ASCII frames (compact)
# -------------------------
SPIN = ["|", "/", "-", "\\"]


def retro_frame(step: int, title: str, inner_w: int = 32) -> str:
    # 5 lines total inside <pre>
    # Keep it compact for phones
    spin = SPIN[step % len(SPIN)]
    pct = min(100, int((step % 24) / 23 * 100))
    bar_w = inner_w - 10
    fill = int(bar_w * pct / 100)
    bar = "[" + "#" * fill + "." * (bar_w - fill) + "]"
    ttl = (title[:inner_w]).ljust(inner_w)
    line2 = f"{spin} {ttl}"
    line3 = f"   {bar} {str(pct).rjust(3)}%"
    line4 = "   " + ("Working on it…" if len("Working on it…") <= inner_w else "Working…").ljust(inner_w)
    line5 = "   " + ("No mercy review mode"[:inner_w]).ljust(inner_w)
    return "\n".join([line2, line3, line4, line5])


async def animate_progress(anchor: Message, chat_id: int, title: str, seconds: float = 2.0) -> Optional[Message]:
    msg: Optional[Message] = None
    start = time.time()
    step = 0

    # create exactly ONE message
    try:
        msg = await anchor.answer(
            html_pre(retro_frame(0, title, inner_w=32)),
            reply_markup=cancel_inline_kb(chat_id),
        )
    except Exception:
        return None

    while time.time() - start < seconds:
        task = RUNNING_TASK.get(chat_id)
        if task and task.cancelled():
            break

        frame = html_pre(retro_frame(step, title, inner_w=32))
        try:
            await msg.edit_text(frame, reply_markup=cancel_inline_kb(chat_id))
        except TelegramBadRequest:
            # don't send duplicates
            break
        except Exception:
            break

        step += 1
        await asyncio.sleep(0.22)

    return msg


# -------------------------
# Figma link -> preview image
# -------------------------
FIGMA_RE = re.compile(r"^https?://(www\.)?figma\.com/(file|design)/[A-Za-z0-9]+/[^?]+(\?.*)?$", re.I)


def looks_like_figma(url: str) -> bool:
    return bool(FIGMA_RE.match(url.strip()))


async def fetch_figma_preview(url: str) -> Optional[bytes]:
    """
    Uses Figma oEmbed to get thumbnail_url, then downloads it.
    Works only if file is public.
    """
    import aiohttp

    oembed = f"https://www.figma.com/api/oembed?url={aiohttp.helpers.quote(url, safe='')}"
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(oembed, headers={"User-Agent": "DesignReviewerBot/1.0"}) as r:
            if r.status != 200:
                return None
            data = await r.json()
            thumb = data.get("thumbnail_url") or data.get("thumbnail_url_with_size")
            if not thumb:
                return None

        async with session.get(thumb, headers={"User-Agent": "DesignReviewerBot/1.0"}) as r2:
            if r2.status != 200:
                return None
            return await r2.read()


# -------------------------
# OCR + annotations
# -------------------------
@dataclass
class TextBox:
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    conf: int


def ocr_boxes(img: Image.Image, lang: str) -> List[TextBox]:
    # Use Tesseract word-level boxes
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    out: List[TextBox] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1
        if not txt:
            continue
        # ignore junk
        if conf < 45:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        out.append(TextBox(text=txt, bbox=(x, y, w, h), conf=conf))
    return out


def ocr_text(img: Image.Image, lang: str) -> str:
    s = pytesseract.image_to_string(img, lang=lang) or ""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def draw_annotations(img: Image.Image, boxes: List[TextBox]) -> Image.Image:
    """
    Draw thin high-contrast rectangles around detected text areas.
    No red; keep neutral (black/white).
    """
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)

    # cluster words into simple lines (by y proximity) to reduce random rectangles
    # This reduces "boxes on no-text" noise.
    boxes_sorted = sorted(boxes, key=lambda b: (b.bbox[1], b.bbox[0]))
    merged: List[Tuple[int, int, int, int]] = []
    y_thresh = 10

    for b in boxes_sorted:
        x, y, w, h = b.bbox
        x2, y2 = x + w, y + h
        placed = False
        for j in range(len(merged)):
            mx1, my1, mx2, my2 = merged[j]
            # same line band
            if abs(y - my1) <= y_thresh or abs(y2 - my2) <= y_thresh:
                # overlap/near by x
                if x <= mx2 + 12 and x2 >= mx1 - 12:
                    merged[j] = (min(mx1, x), min(my1, y), max(mx2, x2), max(my2, y2))
                    placed = True
                    break
        if not placed:
            merged.append((x, y, x2, y2))

    # Draw rectangles
    for (x1, y1, x2, y2) in merged[:60]:
        # outer white, inner black for contrast
        d.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=3)
        d.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=(0, 0, 0), width=1)

    return out


# -------------------------
# ASCII concept safety
# -------------------------
def hard_wrap_lines(text: str, width: int) -> str:
    """
    Hard guarantee: each output line <= width.
    No fancy wrapping — strict cut.
    """
    text = (text or "").replace("\t", " ")
    lines = text.splitlines() if text else []
    out: List[str] = []

    for ln in lines:
        ln = ln.rstrip("\n\r")
        if not ln:
            out.append("")
            continue
        # Normalize any crazy long tokens by cutting
        while len(ln) > width:
            out.append(ln[:width])
            ln = ln[width:]
        out.append(ln[:width])

    # Secondary clamp (paranoia)
    out = [ln[:width] for ln in out]
    return "\n".join(out).strip("\n")


# -------------------------
# LLM review (plain text markers)
# -------------------------
def have_llm() -> bool:
    return bool(LLM_ENABLED and OpenAI_API_KEY and OpenAI is not None)


def make_llm_client():
    return OpenAI(api_key=OPENAI_API_KEY)  # type: ignore


def llm_prompt(chat_id: int) -> str:
    L = lang_of(chat_id)
    if L == "EN":
        return (
            "You are a tough but fair senior product designer doing a design review.\n"
            "Be honest and nitpicky, but no profanity.\n"
            "Use only monochrome symbols (no colorful emoji).\n"
            "Do not output JSON.\n\n"
            "Return exactly in this format with these markers:\n"
            "===WHAT_I_SEE===\n"
            "(short paragraph)\n"
            "===VERDICT===\n"
            "Score: X/10\n"
            "Praise: (1-2 bullets)\n"
            "Issues: (3-7 bullets)\n"
            "Fixes: (3-7 bullets)\n"
            "===ASCII_CONCEPT===\n"
            "(ASCII concept only)\n\n"
            "ASCII concept rules:\n"
            f"- Max {ASCII_CONCEPT_COLS} characters per line.\n"
            "- No long unbroken tokens; use spaces.\n"
            "- Use only ASCII characters: letters, digits, spaces, .,:;_-+=|/\\[]()#*\n"
        )
    else:
        return (
            "Ты — жёсткий, но справедливый сеньор продуктовый дизайнер. Делаешь дизайн-ревью.\n"
            "Честно докапывайся, но без мата.\n"
            "Используй только монохромные символы (без цветных эмодзи).\n"
            "Не выводи JSON.\n\n"
            "Верни строго в таком формате с маркерами:\n"
            "===WHAT_I_SEE===\n"
            "(короткий абзац)\n"
            "===VERDICT===\n"
            "Оценка: X/10\n"
            "Похвала: (1-2 пункта)\n"
            "Проблемы: (3-7 пунктов)\n"
            "Что сделать: (3-7 пунктов)\n"
            "===ASCII_CONCEPT===\n"
            "(только ASCII-концепт)\n\n"
            "Правила ASCII-концепта:\n"
            f"- Максимум {ASCII_CONCEPT_COLS} символа в строке.\n"
            "- Без длинных слов без пробелов.\n"
            "- Только ASCII-символы: буквы, цифры, пробелы, .,:;_-+=|/\\[]()#*\n"
        )


def parse_llm(text: str) -> Tuple[str, str, str]:
    """
    Returns: what_i_see, verdict, ascii_concept
    """
    text = text or ""
    what = ""
    verdict = ""
    concept = ""
    m1 = re.search(r"===WHAT_I_SEE===\s*(.*?)\s*===VERDICT===", text, re.S)
    m2 = re.search(r"===VERDICT===\s*(.*?)\s*===ASCII_CONCEPT===", text, re.S)
    m3 = re.search(r"===ASCII_CONCEPT===\s*(.*)\s*$", text, re.S)
    if m1:
        what = m1.group(1).strip()
    if m2:
        verdict = m2.group(1).strip()
    if m3:
        concept = m3.group(1).strip()
    return what, verdict, concept


async def llm_review(chat_id: int, img_bytes: bytes, ocr_plain: str) -> Tuple[str, str, str]:
    """
    Calls OpenAI Responses API with image + OCR text.
    Returns parsed sections.
    """
    client = make_llm_client()

    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    prompt = llm_prompt(chat_id)
    user_ctx = (
        "Context:\n"
        "- The user sent a UI screenshot or a Figma frame preview.\n"
        "- OCR text extracted (may be noisy):\n"
        f"{ocr_plain[:2500]}\n"
    )

    # Responses API expects content item types like: input_text, input_image
    resp = await asyncio.to_thread(
        client.responses.create,
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_ctx},
                    {"type": "input_image", "image_url": data_url},
                    {
                        "type": "input_text",
                        "text": "Now produce the review with the required markers. Keep it compact.",
                    },
                ],
            },
        ],
    )

    # Extract output text
    out_text = ""
    try:
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for c in item.content:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        out_text += getattr(c, "text", "") or ""
    except Exception:
        out_text = getattr(resp, "output_text", "") or ""

    what, verdict, concept = parse_llm(out_text)

    # Hard safety for concept
    concept = concept.replace("\r", "")
    concept = hard_wrap_lines(concept, ASCII_CONCEPT_COLS)

    if not what or not verdict or not concept:
        raise ValueError("LLM returned incomplete sections")

    return what, verdict, concept


# -------------------------
# Minimal fallback (if LLM fails)
# -------------------------
def fallback_review(chat_id: int, ocr_plain: str) -> Tuple[str, str, str]:
    L = lang_of(chat_id)
    if L == "EN":
        what = "A UI screen with text elements (OCR extracted)."
        verdict = (
            "Score: 5/10\n"
            "Praise:\n"
            "- At least the flow exists.\n"
            "Issues:\n"
            "- OCR is noisy; screenshot may be too small.\n"
            "- Copy might be unclear or too formal.\n"
            "Fixes:\n"
            "- Send a larger / clearer screenshot.\n"
            "- Shorten copy and make actions explicit.\n"
        )
        concept = (
            "+------------------------------+\n"
            "| Title                         |\n"
            "| Short help text               |\n"
            "|                               |\n"
            "| [ Primary action ]            |\n"
            "| [ Secondary ]                 |\n"
            "+------------------------------+\n"
        )
    else:
        what = "Экран интерфейса с текстовыми элементами (OCR извлёк текст)."
        verdict = (
            "Оценка: 5/10\n"
            "Похвала:\n"
            "- Сценарий хотя бы читается.\n"
            "Проблемы:\n"
            "- OCR шумный — скрин, похоже, мелкий.\n"
            "- Текст может быть туманным/канцелярским.\n"
            "Что сделать:\n"
            "- Пришли скрин крупнее/чётче.\n"
            "- Укорачивай формулировки, делай действия явными.\n"
        )
        concept = (
            "+------------------------------+\n"
            "| Заголовок                     |\n"
            "| Короткая подсказка            |\n"
            "|                               |\n"
            "| [ Основное действие ]         |\n"
            "| [ Дополнительно ]             |\n"
            "+------------------------------+\n"
        )

    return what, verdict, hard_wrap_lines(concept, ASCII_CONCEPT_COLS)


# -------------------------
# Router / handlers
# -------------------------
router = Router()


@router.message(Command("start"))
async def on_start(m: Message):
    chat_id = m.chat.id
    LANG.setdefault(chat_id, "EN")
    await m.answer(t(chat_id, "start"), reply_markup=main_menu_kb(chat_id))


@router.callback_query(F.data == "cancel_review")
async def on_cancel(cb: CallbackQuery):
    chat_id = cb.message.chat.id if cb.message else cb.from_user.id
    task = RUNNING_TASK.get(chat_id)
    if task and not task.done():
        task.cancel()
    try:
        await cb.answer("OK")
    except Exception:
        pass
    if cb.message:
        await cb.message.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))


@router.message(F.text)
async def on_text(m: Message):
    chat_id = m.chat.id
    txt = (m.text or "").strip()

    # Channel button
    if txt in ("Product design channel", "Канал о продуктовом дизайне"):
        await m.answer(f"{CHANNEL_HANDLE}\n{CHANNEL_URL}", reply_markup=main_menu_kb(chat_id))
        return

    # Language toggle buttons
    if txt in (TEXT["EN"]["menu_lang_ru"], TEXT["RU"]["menu_lang_ru"]):
        LANG[chat_id] = "RU"
        await m.answer(t(chat_id, "lang_switched_ru"), reply_markup=main_menu_kb(chat_id))
        return
    if txt in (TEXT["EN"]["menu_lang_en"], TEXT["RU"]["menu_lang_en"]):
        LANG[chat_id] = "EN"
        await m.answer(t(chat_id, "lang_switched_en"), reply_markup=main_menu_kb(chat_id))
        return

    # How it works
    if txt == t(chat_id, "menu_how"):
        await m.answer(t(chat_id, "how"), reply_markup=main_menu_kb(chat_id))
        return

    # "Send for review" just hints
    if txt == t(chat_id, "menu_review"):
        hint = (
            "Send a screenshot or a public Figma frame link."
            if lang_of(chat_id) == "EN"
            else "Кидай скриншот или публичную ссылку на фрейм Figma."
        )
        await m.answer(hint, reply_markup=ReplyKeyboardRemove())
        return

    # Figma link
    if looks_like_figma(txt):
        await start_review_from_figma(m, txt)
        return

    # Unknown text
    await m.answer(t(chat_id, "bad_link"), reply_markup=main_menu_kb(chat_id))


@router.message(F.photo)
async def on_photo(m: Message):
    # pick highest resolution
    chat_id = m.chat.id
    ph = m.photo[-1]
    file = await m.bot.get_file(ph.file_id)
    img_bytes = await m.bot.download_file(file.file_path)
    b = img_bytes.read() if hasattr(img_bytes, "read") else img_bytes  # type: ignore
    await start_review_from_image(m, b)


# -------------------------
# Review flows
# -------------------------
async def start_review_from_figma(m: Message, url: str):
    chat_id = m.chat.id

    # cancel previous
    prev = RUNNING_TASK.get(chat_id)
    if prev and not prev.done():
        prev.cancel()

    async def task():
        try:
            await m.answer(" ", reply_markup=ReplyKeyboardRemove())
            await animate_progress(m, chat_id, title="Fetching preview…" if lang_of(chat_id) == "EN" else "Забираю превью…", seconds=1.4)

            img = await fetch_figma_preview(url)
            if not img:
                await m.answer(t(chat_id, "figma_private"), reply_markup=main_menu_kb(chat_id))
                return

            # Show preview first? (requested)
            try:
                await m.answer_photo(img, caption="Preview" if lang_of(chat_id) == "EN" else "Превью")
            except Exception:
                pass

            await process_image_and_reply(m, img)

        except asyncio.CancelledError:
            await m.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))
        except Exception:
            await m.answer(t(chat_id, "llm_fail"), reply_markup=main_menu_kb(chat_id))

    RUNNING_TASK[chat_id] = asyncio.create_task(task())


async def start_review_from_image(m: Message, img_bytes: bytes):
    chat_id = m.chat.id

    prev = RUNNING_TASK.get(chat_id)
    if prev and not prev.done():
        prev.cancel()

    async def task():
        try:
            await m.answer(" ", reply_markup=ReplyKeyboardRemove())
            await animate_progress(m, chat_id, title="Loading…" if lang_of(chat_id) == "EN" else "Загружаю…", seconds=1.2)
            await process_image_and_reply(m, img_bytes)
        except asyncio.CancelledError:
            await m.answer(t(chat_id, "cancelled"), reply_markup=main_menu_kb(chat_id))
        except Exception:
            await m.answer(t(chat_id, "llm_fail"), reply_markup=main_menu_kb(chat_id))

    RUNNING_TASK[chat_id] = asyncio.create_task(task())


async def process_image_and_reply(anchor: Message, img_bytes: bytes):
    chat_id = anchor.chat.id

    # processing animation (single message, edited)
    await animate_progress(anchor, chat_id, title=t(chat_id, "processing"), seconds=2.0)

    # Decode image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # OCR
    boxes = ocr_boxes(img, OCR_LANG)
    plain = ocr_text(img, OCR_LANG)

    # LLM
    if have_llm():
        try:
            what, verdict, concept = await llm_review(chat_id, img_bytes, plain)
        except Exception:
            what, verdict, concept = fallback_review(chat_id, plain)
    else:
        what, verdict, concept = fallback_review(chat_id, plain)

    # Message 1: What I see
    await anchor.answer(f"{t(chat_id,'what_i_see_title')}\n{html_escape(what)}", reply_markup=None)

    # Message 2: Verdict + recommendations (keep it left-aligned, no weird quotes)
    await anchor.answer(f"{t(chat_id,'verdict_title')}\n{html_escape(verdict)}", reply_markup=None)

    # Message 3: Annotated screenshot
    anno = draw_annotations(img, boxes)
    buf = io.BytesIO()
    anno.save(buf, format="PNG")
    buf.seek(0)
    await anchor.answer(t(chat_id, "anno_title"), reply_markup=None)
    try:
        await anchor.answer_photo(buf.getvalue())
    except Exception:
        # fallback as document if needed
        try:
            await anchor.answer_document(buf.getvalue())
        except Exception:
            pass

    # Progress before concept
    await animate_progress(anchor, chat_id, title=t(chat_id, "processing2"), seconds=1.4)

    # Message 4: ASCII concept (monospace)
    await anchor.answer(t(chat_id, "concept_title"), reply_markup=None)
    await anchor.answer(html_pre(concept), reply_markup=main_menu_kb(chat_id))

    # Done — restore menu (only here)
    # (Also keep RUNNING_TASK clean)
    RUNNING_TASK.pop(chat_id, None)


# -------------------------
# Main
# -------------------------
async def main():
    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())