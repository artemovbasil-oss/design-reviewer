# bot.py — Design Review Partner (aiogram 3.7.0)
# - ASCII прогресс с fallback (если edit нельзя, создаём 1 новый прогресс-месседж и редактируем его)
# - Hybrid: OCR (если доступен) -> иначе LLM "extract"
# - 3 итоговых сообщения: что вижу / визуал (оценка) / тексты
# - Без тех. деталей (px/цвет-коды), шрифты/палитра — только догадки

import os
import re
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List

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
BTN_SEND = "🖼 Закинуть скрин"
BTN_HELP = "ℹ️ Как пользоваться"
BTN_PING = "🏓 Ping"

keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_SEND)],
        [KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_PING)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Кидай скрин — я разберу его по-взрослому.",
)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

# per-chat lock (чтобы прогресс/ответы не путались)
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


def ascii_bar(i: int) -> str:
    frames = [
        "▱▱▱▱▱",
        "▰▱▱▱▱",
        "▰▰▱▱▱",
        "▰▰▰▱▱",
        "▰▰▰▰▱",
        "▰▰▰▰▰",
        "▰▰▰▰▰✓",
    ]
    return frames[max(0, min(i, len(frames) - 1))]


def spinner(i: int) -> str:
    return ["|", "/", "—", "\\"][i % 4]


def parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# =============================
# Progress animation (no-spam but always visible)
# =============================
async def safe_edit_text_or_recreate(msg: Message, text: str) -> Message:
    """
    Пытаемся отредактировать msg.
    Если Telegram не даёт редактировать (message can't be edited) —
    создаём ОДНО новое прогресс-сообщение и продолжаем на нём.
    """
    try:
        await msg.edit_text(text)
        return msg
    except TelegramBadRequest:
        try:
            new_msg = await msg.answer(text)
            return new_msg
        except TelegramBadRequest:
            return msg


async def animate_progress(msg: Message, title: str = "🔍 Смотрю внимательно…") -> Message:
    current = msg
    for i in range(10):
        bar = ascii_bar(min(i, 6))
        frame = f"{title} {spinner(i)}\n<pre>{bar}</pre>"
        current = await safe_edit_text_or_recreate(current, frame)
        await asyncio.sleep(0.22)
    return current


async def set_progress(msg: Message, title: str, step: int) -> Message:
    bar = ascii_bar(step)
    frame = f"{title} {spinner(step)}\n<pre>{bar}</pre>"
    return await safe_edit_text_or_recreate(msg, frame)


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
    """
    Returns:
    {
      ok: bool,
      text: str,
      blocks: [ {text, kind_guess} ... ]  # bbox intentionally removed for simplicity
    }
    """
    if not OCR_PY_AVAILABLE:
        return {"ok": False, "reason": "pytesseract not installed", "text": "", "blocks": []}

    try:
        img = preprocess_for_ocr(pil)
        data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    except Exception as e:
        # Most common on Railway: tesseract binary missing
        return {"ok": False, "reason": f"tesseract error: {e}", "text": "", "blocks": []}

    n = len(data.get("text", []))
    lines: Dict[str, List[str]] = {}
    full_words: List[str] = []

    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf = -1.0
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            pass
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
        # Very rough guess
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
    """
    Fallback when OCR isn't available: ask LLM to extract text blocks.
    Returns:
      { ok: bool, text: str, blocks: [{text, kind_guess}] }
    """
    prompt = """
Ты видишь скрин интерфейса. Твоя задача — вытащить текст и понять структуру.
Верни СТРОГО JSON:
{
  "text": "весь текст на экране одной строкой (если что-то не читается — пропусти)",
  "blocks": [
    {"text":"...", "kind_guess":"title_or_button|button_like|text|hint|status"},
    ...
  ]
}
Без лишних ключей. Без пояснений. Только JSON.
"""
    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_base64": image_b64},
                ],
            }
        ],
        max_output_tokens=700,
    )

    out_text = ""
    for item in getattr(resp, "output", []) or []:
        for c in item.content or []:
            if getattr(c, "type", None) == "output_text":
                out_text += getattr(c, "text", "") + "\n"

    data = parse_llm_json(out_text.strip())
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
- Никаких технических деталей (пиксели, коды цветов, расчёты).
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
  "visual": "5–12 пунктов: визуал/UX (с похвалой, если есть)",
  "text": "6–14 пунктов: текст (каждый пункт: Проблема → Почему плохо → Как исправить)"
}}
"""

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_base64": image_b64},
                ],
            }
        ],
        max_output_tokens=950,
    )

    out_text = ""
    for item in getattr(resp, "output", []) or []:
        for c in item.content or []:
            if getattr(c, "type", None) == "output_text":
                out_text += getattr(c, "text", "") + "\n"

    out_text = out_text.strip()
    data = parse_llm_json(out_text)
    if not data:
        # fallback: plain text without dict junk
        return {
            "description": (out_text[:900] or "Не смог собрать отчёт из ответа модели."),
            "score": 5,
            "visual": "—",
            "text": "—",
        }

    return {
        "description": str(data.get("description", "")).strip(),
        "score": clamp_score(data.get("score", 6)),
        "visual": str(data.get("visual", "")).strip(),
        "text": str(data.get("text", "")).strip(),
    }


# =============================
# Handlers
# =============================
@dp.message(F.text.in_({"/start", "start"}))
async def start(m: Message):
    await m.answer(
        "👋 Я — твой <b>партнёр по дизайн-ревью</b>.\n\n"
        "Кидай скрин интерфейса — я:\n"
        "1) скажу, что вижу\n"
        "2) разнесу (или похвалю) визуал\n"
        "3) разнесу (или похвалю) тексты\n\n"
        "Жми кнопку снизу или просто отправь картинку.",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_HELP)
async def help_msg(m: Message):
    await m.answer(
        "Как пользоваться:\n"
        "• Отправь скриншот.\n"
        "• Я покажу прогресс ASCII.\n"
        "• Потом пришлю 3 сообщения: описание / визуал / тексты.\n\n"
        "Если текст мелкий — пришли скрин крупнее (или обрежь лишнее) — будет точнее.",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_PING)
async def ping(m: Message):
    await m.answer(
        f"pong ✅\nMODEL: <code>{html_escape(LLM_MODEL)}</code>\nOCR(py): <code>{'on' if OCR_PY_AVAILABLE else 'off'}</code>",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_SEND)
async def ask(m: Message):
    await m.answer("Ок. Закидывай скрин. Посмотрю как следует.", reply_markup=keyboard)


@dp.message(F.photo)
async def handle_photo(m: Message):
    chat_id = m.chat.id
    lock = get_chat_lock(chat_id)

    if lock.locked():
        await m.answer(
            "⛔ Я уже разбираю другой скрин.\n"
            "Кинь этот чуть позже, иначе мы сами себе всё перемешаем.",
            reply_markup=keyboard,
        )
        return

    async with lock:
        # 1) Initial progress (без клавиатуры — меньше шансов, что Telegram запретит edit)
        progress = await m.answer("⏳ Принял. Загружаю…")
        progress = await animate_progress(progress, title="🔍 Смотрю внимательно…")

        photo = m.photo[-1]
        file = await bot.get_file(photo.file_id)

        bio = BytesIO()
        await bot.download_file(file.file_path, destination=bio)
        bio.seek(0)

        try:
            img = Image.open(bio).convert("RGBA")
        except Exception:
            await m.answer("⚠️ Не смог открыть картинку. Пришли другой файл.", reply_markup=keyboard)
            return

        # 2) Upscale small images (helps both OCR and vision)
        w, h = img.size
        if max(w, h) < 1400:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)

        img_b64 = img_to_base64_png(img)

        # 3) Extract text/structure (OCR first)
        progress = await set_progress(progress, "🧾 Читаю текст…", 3)

        extracted = {"ok": False, "text": "", "blocks": []}
        ocr = ocr_extract(img)
        if ocr.get("ok") and (len((ocr.get("text") or "").strip()) >= 12):
            extracted = {"ok": True, "text": ocr.get("text", ""), "blocks": ocr.get("blocks", [])}
        else:
            # LLM extract fallback
            extracted = llm_extract_text_structure(img_b64)
            if not extracted.get("ok"):
                # last resort: keep what OCR gave (even if weak)
                extracted = {"ok": False, "text": ocr.get("text", ""), "blocks": ocr.get("blocks", [])}

        # 4) Review
        progress = await set_progress(progress, "🧠 Думаю…", 5)

        try:
            result = analyze_ui_with_openai(img_b64, extracted)
        except Exception:
            await m.answer(
                "⚠️ Упал на анализе.\n\n"
                "Скорее всего:\n"
                "• слишком мелкий текст\n"
                "• экран перегружен\n"
                "• часть интерфейса обрезана\n\n"
                "Попробуй:\n"
                "— скрин крупнее\n"
                "— обрезать лишнее вокруг\n"
                "— на вебе: зум 125–150% и переснять",
                reply_markup=keyboard,
            )
            return

        progress = await set_progress(progress, "✅ Готово.", 6)

        desc = html_escape(result.get("description", "")) or "—"
        visual = html_escape(result.get("visual", "")) or "—"
        text = html_escape(result.get("text", "")) or "—"
        score = clamp_score(result.get("score", 6))

        # 5) Final 3 messages (+ keyboard again)
        await m.answer(f"👀 <b>Что я вижу</b>\n{desc}", reply_markup=keyboard)
        await m.answer(f"🎛 <b>Визуал</b> — оценка: <b>{score}/10</b>\n{visual}", reply_markup=keyboard)
        await m.answer(f"✍️ <b>Тексты</b>\n{text}", reply_markup=keyboard)


@dp.message()
async def fallback(m: Message):
    await m.answer(
        "Я жду скриншот интерфейса.\n"
        "Отправь картинку — и я устрою ревью.",
        reply_markup=keyboard,
    )


async def main():
    print(f"✅ Design Review Partner starting… model={LLM_MODEL}, OCR_PY={OCR_PY_AVAILABLE}, CV={CV_AVAILABLE}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
