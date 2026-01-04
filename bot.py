# bot.py — Design Review Partner (aiogram 3.7.0)
# Style update:
# - emojis: monochrome-ish only (minimal)
# - progress: retro ASCII
# - output normalization: lists -> bullet text, strip ["'..."]

import os
import re
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from PIL import Image

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest

from openai import OpenAI

# OCR (optional)
try:
    import pytesseract  # type: ignore
    OCR_PY_AVAILABLE = True
except Exception:
    OCR_PY_AVAILABLE = False

# optional OpenCV for better OCR
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    CV_AVAILABLE = True
except Exception:
    CV_AVAILABLE = False


# =============================
# Local .env loader (no python-dotenv)
# =============================
def load_local_env_file() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
    except Exception:
        pass


load_local_env_file()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
OCR_LANG = os.getenv("OCR_LANG", "rus+eng").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set (Railway Variables or local .env)")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set (Railway Variables or local .env)")

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# Telegram UI
# =============================
BTN_SEND = "◼︎ Отправить скрин"
BTN_HELP = "◻︎ Как пользоваться"
BTN_PING = "▶︎ Ping"

keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_SEND)],
        [KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_PING)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Кидай скрин — разберём по-взрослому.",
)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

# per-chat lock (to avoid mixed replies)
_CHAT_LOCKS: Dict[int, asyncio.Lock] = {}


def get_chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _CHAT_LOCKS[chat_id] = lock
    return lock


# =============================
# Helpers
# =============================
def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def clamp_score(x: Any) -> int:
    try:
        n = int(x)
    except Exception:
        n = 6
    return max(1, min(10, n))


def img_to_base64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def data_url_from_b64_png(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


def extract_output_text(resp: Any) -> str:
    out_text = ""
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                out_text += getattr(c, "text", "") + "\n"
    return out_text.strip()


def parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def strip_listish_wrappers(s: str) -> str:
    """
    Remove ugly wrappers like "['...']" or '["..."]' if LLM returns stringified list.
    """
    s = (s or "").strip()
    # common: "['a', 'b']" or '["a","b"]'
    if (s.startswith("[") and s.endswith("]")) and (("'" in s) or ('"' in s)):
        # try parse as python-ish list by converting quotes -> json safely-ish
        # easiest: attempt json after minor fixes
        candidate = s
        # if single quotes used, convert to double quotes cautiously
        if "'" in candidate and '"' not in candidate:
            candidate = candidate.replace("'", '"')
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return "\n".join([str(x).strip() for x in arr if str(x).strip()])
        except Exception:
            pass
    return s


def bullets_from_any(x: Any, bullet: str = "• ") -> str:
    """
    If x is list -> bullet lines.
    If x is string -> clean wrappers.
    """
    if isinstance(x, list):
        items = []
        for it in x:
            t = str(it).strip()
            if t:
                items.append(f"{bullet}{t}")
        return "\n".join(items) if items else "—"
    if isinstance(x, dict):
        # shouldn't happen, but keep readable
        return "\n".join([f"{bullet}{k}: {v}" for k, v in x.items()]) or "—"
    s = strip_listish_wrappers(str(x or "").strip())
    return s or "—"


# =============================
# Retro ASCII progress
# =============================
def retro_bar(step: int, total: int = 12) -> str:
    step = max(0, min(step, total))
    filled = "#" * step
    empty = "." * (total - step)
    return f"[{filled}{empty}]"


def retro_spinner(i: int) -> str:
    return ["|", "/", "-", "\\"][i % 4]


def retro_screen(title: str, i: int, step: int) -> str:
    bar = retro_bar(step)
    spin = retro_spinner(i)
    # little retro "scanline" vibe
    lines = [
        f"{title} {spin}",
        bar,
        "--------------------",
        "SIGNAL: OK   MODE: SCAN",
    ]
    return "<pre>" + "\n".join(lines) + "</pre>"


async def safe_edit_text_or_recreate(msg: Message, text: str) -> Message:
    try:
        await msg.edit_text(text)
        return msg
    except TelegramBadRequest:
        try:
            new_msg = await msg.answer(text)
            return new_msg
        except TelegramBadRequest:
            return msg


async def animate_progress(msg: Message, title: str = "SCAN") -> Message:
    current = msg
    # 2 phases: load + scan
    for i in range(8):
        current = await safe_edit_text_or_recreate(current, retro_screen(title, i, step=min(i, 6)))
        await asyncio.sleep(0.18)
    for i in range(8, 16):
        current = await safe_edit_text_or_recreate(current, retro_screen(title, i, step=min(i - 8, 6)))
        await asyncio.sleep(0.18)
    return current


async def set_progress(msg: Message, title: str, i: int, step: int) -> Message:
    return await safe_edit_text_or_recreate(msg, retro_screen(title, i, step))


# =============================
# OCR pipeline (best effort)
# =============================
def preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    if not CV_AVAILABLE:
        return pil.convert("RGB")

    img = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    return Image.fromarray(thr)


def ocr_extract(pil: Image.Image) -> Dict[str, Any]:
    if not OCR_PY_AVAILABLE:
        return {"ok": False, "reason": "pytesseract not installed", "text": "", "blocks": []}

    try:
        img = preprocess_for_ocr(pil)
        data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    except Exception as e:
        return {"ok": False, "reason": f"tesseract error: {e}", "text": "", "blocks": []}

    n = len(data.get("text", []))
    lines: Dict[str, List[str]] = {}
    full_words: List[str] = []

    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0
        if conf >= 0 and conf < 35:
            continue

        full_words.append(txt)
        key = f"{data.get('block_num',[0])[i]}:{data.get('par_num',[0])[i]}:{data.get('line_num',[0])[i]}"
        lines.setdefault(key, []).append(txt)

    blocks = []
    for _, words in lines.items():
        line = " ".join(words).strip()
        if not line:
            continue
        kind = "text"
        if len(line) <= 20:
            kind = "title_or_button"
        if len(line) <= 14:
            kind = "button_like"
        blocks.append({"text": line, "kind_guess": kind})

    return {"ok": True, "text": " ".join(full_words).strip(), "blocks": blocks}


# =============================
# LLM: extract (fallback when OCR fails)
# =============================
def llm_extract_text_structure(image_b64: str) -> Dict[str, Any]:
    prompt = """
Ты видишь скрин интерфейса. Твоя задача — вытащить текст и структуру.
Верни СТРОГО JSON:
{
  "text": "весь текст на экране одной строкой (если что-то не читается — пропусти)",
  "blocks": [
    {"text":"...", "kind_guess":"title_or_button|button_like|text|hint|status"},
    ...
  ]
}
Только JSON. Без пояснений.
""".strip()

    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url_from_b64_png(image_b64)},
                    ],
                }
            ],
            max_output_tokens=700,
        )
    except Exception as e:
        return {"ok": False, "reason": f"LLM extract error: {e}", "text": "", "blocks": []}

    out_text = extract_output_text(resp)
    data = parse_llm_json(out_text)
    if not data:
        return {"ok": False, "text": "", "blocks": []}

    text = str(data.get("text", "")).strip()
    blocks = data.get("blocks", [])
    if not isinstance(blocks, list):
        blocks = []

    cleaned = []
    for b in blocks[:80]:
        if not isinstance(b, dict):
            continue
        t = str(b.get("text", "")).strip()
        if not t:
            continue
        k = str(b.get("kind_guess", "text")).strip()
        cleaned.append({"text": t, "kind_guess": k})

    return {"ok": True, "text": text, "blocks": cleaned}


# =============================
# LLM: review (image + extracted text)
# =============================
def analyze_ui_with_openai(image_b64: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
    ocr_text = (extracted.get("text") or "").strip()
    blocks = extracted.get("blocks") or []
    blocks_short = blocks[:80]

    prompt = f"""
Ты — старший продуктовый дизайнер и требовательный дизайн-ревьюер.
Говоришь по-русски. Без мата. Без сюсюканья.
Если хорошо — хвали конкретно. Если плохо — ругай конкретно и предлагай улучшения.

Ограничения:
- Никаких технических деталей (пиксели, коды цветов, измерения).
- Про шрифт/палитру — только предположения ("похоже на sans-serif типа Inter/SF/Roboto").
- Не путай заголовки и кнопки. Сверяйся с картинкой и блоками текста.
- Не выдумывай элементы.

Извлечённый текст:
{ocr_text[:2000]}

Блоки (строки) с грубым guess:
{json.dumps(blocks_short, ensure_ascii=False)}

Верни СТРОГО JSON:
{{
  "description": "2–6 предложений: что происходит на экране",
  "score": 1-10,
  "visual": ["5–12 пунктов: визуал/UX (с похвалой, если есть)"],
  "text": ["6–14 пунктов: текст (каждый пункт: Проблема → Почему плохо → Как исправить)"]
}}
""".strip()

    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url_from_b64_png(image_b64)},
                    ],
                }
            ],
            max_output_tokens=950,
        )
    except Exception as e:
        return {
            "description": "Не смог вызвать модель (ошибка на стороне API).",
            "score": 5,
            "visual": [f"Причина: {e}"],
            "text": ["Проверь ключ/модель. Нужно, чтобы модель умела работать с картинками."],
        }

    out_text = extract_output_text(resp)
    data = parse_llm_json(out_text)
    if not data:
        return {
            "description": (out_text[:900] or "Не смог собрать отчёт из ответа модели."),
            "score": 5,
            "visual": ["—"],
            "text": ["—"],
        }

    return {
        "description": str(data.get("description", "")).strip(),
        "score": clamp_score(data.get("score", 6)),
        "visual": data.get("visual", "—"),
        "text": data.get("text", "—"),
    }


# =============================
# Handlers
# =============================
@dp.message(F.text.in_({"/start", "start"}))
async def start(m: Message):
    await m.answer(
        "<b>Партнёр дизайн-ревью</b>\n\n"
        "Кидай скрин интерфейса — я:\n"
        "• скажу, что вижу\n"
        "• разберу визуал и UX (могу похвалить, но и докопаюсь)\n"
        "• разберу тексты (что не так и как поправить)\n\n"
        "Жми кнопку снизу или просто отправь картинку.",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_HELP)
async def help_msg(m: Message):
    await m.answer(
        "Как пользоваться:\n"
        "1) Отправь скриншот.\n"
        "2) Я покажу ретро-прогресс.\n"
        "3) Потом 3 сообщения: описание / визуал / тексты.\n\n"
        "Если текст мелкий — пришли скрин крупнее или обрежь лишнее.",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_PING)
async def ping(m: Message):
    await m.answer(
        f"pong\nMODEL: <code>{html_escape(LLM_MODEL)}</code>\nOCR: <code>{'on' if OCR_PY_AVAILABLE else 'off'}</code>",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_SEND)
async def ask(m: Message):
    await m.answer("Ок. Кидай скрин. Посмотрю внимательно.", reply_markup=keyboard)


@dp.message(F.photo)
async def handle_photo(m: Message):
    chat_id = m.chat.id
    lock = get_chat_lock(chat_id)

    if lock.locked():
        await m.answer(
            "Секунду. Я уже разбираю другой скрин.\n"
            "Кинь этот сразу после — иначе смешаем отчёты.",
            reply_markup=keyboard,
        )
        return

    async with lock:
        progress = await m.answer("SCAN\n<pre>[............]</pre>")
        progress = await animate_progress(progress, title="SCAN")

        photo = m.photo[-1]
        file = await bot.get_file(photo.file_id)

        bio = BytesIO()
        await bot.download_file(file.file_path, destination=bio)
        bio.seek(0)

        try:
            img = Image.open(bio).convert("RGBA")
        except Exception:
            await m.answer("Не смог открыть картинку. Пришли другой файл.", reply_markup=keyboard)
            return

        # Upscale small images (helps OCR + vision)
        w, h = img.size
        if max(w, h) < 1400:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)

        img_b64 = img_to_base64_png(img)

        # Extract text/structure
        progress = await set_progress(progress, "OCR", i=1, step=3)

        extracted = {"ok": False, "text": "", "blocks": []}
        ocr = ocr_extract(img)

        if ocr.get("ok") and len((ocr.get("text") or "").strip()) >= 12:
            extracted = {"ok": True, "text": ocr.get("text", ""), "blocks": ocr.get("blocks", [])}
        else:
            extracted = llm_extract_text_structure(img_b64)
            if not extracted.get("ok"):
                extracted = {"ok": False, "text": ocr.get("text", ""), "blocks": ocr.get("blocks", [])}

        progress = await set_progress(progress, "REVIEW", i=2, step=6)

        result = analyze_ui_with_openai(img_b64, extracted)

        progress = await set_progress(progress, "DONE", i=3, step=12)

        desc = html_escape(str(result.get("description", "")).strip()) or "—"
        score = clamp_score(result.get("score", 6))

        visual_txt = bullets_from_any(result.get("visual", "—"), bullet="• ")
        text_txt = bullets_from_any(result.get("text", "—"), bullet="• ")

        visual_txt = html_escape(visual_txt)
        text_txt = html_escape(text_txt)

        await m.answer(f"<b>Что вижу</b>\n{desc}", reply_markup=keyboard)
        await m.answer(f"<b>Визуал</b> — оценка: <b>{score}/10</b>\n{visual_txt}", reply_markup=keyboard)
        await m.answer(f"<b>Тексты</b>\n{text_txt}", reply_markup=keyboard)


@dp.message()
async def fallback(m: Message):
    await m.answer(
        "Я жду скрин интерфейса.\nОтправь картинку — и я устрою ревью.",
        reply_markup=keyboard,
    )


async def main():
    print(f"Design Review starting… model={LLM_MODEL}, OCR={OCR_PY_AVAILABLE}, CV={CV_AVAILABLE}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
