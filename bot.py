import os
import re
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from PIL import Image

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest

from openai import OpenAI

# OCR (optional)
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# optional OpenCV for better OCR
try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except Exception:
    CV_AVAILABLE = False


# =============================
# Optional local .env loader (NO python-dotenv dependency)
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
    input_field_placeholder="Кидай скрин — разберу по делу, без сюсюканья.",
)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

# per-chat lock
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


def ascii_frame(i: int) -> str:
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


def spinner_frame(i: int) -> str:
    sp = ["|", "/", "—", "\\"]
    return sp[i % len(sp)]


async def safe_edit_text(msg: Message, text: str) -> bool:
    try:
        await msg.edit_text(text)
        return True
    except TelegramBadRequest:
        return False


async def animate_progress(msg: Message, title: str = "🔍 Смотрю внимательно…") -> None:
    for i in range(6):
        ok = await safe_edit_text(msg, f"{title} {spinner_frame(i)}\n<code>{ascii_frame(i)}</code>")
        if not ok:
            break
        await asyncio.sleep(0.22)


async def progress_set(msg: Message, title: str, i: int) -> None:
    await safe_edit_text(msg, f"{title} {spinner_frame(i)}\n<code>{ascii_frame(i)}</code>")


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
# OCR pipeline (hybrid restore)
# =============================
def preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    """
    Улучшает контраст/читаемость для OCR.
    Если нет OpenCV — возвращает как есть.
    """
    if not CV_AVAILABLE:
        return pil.convert("RGB")

    img = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # лёгкая нормализация
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # адаптивный порог
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 8)
    return Image.fromarray(thr)


def ocr_extract(pil: Image.Image) -> Dict[str, Any]:
    """
    Возвращает:
    {
      "ok": bool,
      "text": str,
      "blocks": [ { "text": str, "bbox": [x,y,w,h], "kind_guess": str } ... ]
    }
    """
    if not OCR_AVAILABLE:
        return {"ok": False, "reason": "pytesseract not installed", "text": "", "blocks": []}

    # если tesseract binary отсутствует — pytesseract кинет ошибку
    try:
        img = preprocess_for_ocr(pil)
        # data gives word-level boxes; we'll aggregate to line-ish blocks using 'line_num'
        data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    except Exception as e:
        return {"ok": False, "reason": f"tesseract error: {e}", "text": "", "blocks": []}

    n = len(data.get("text", []))
    blocks_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    full_text_parts: List[str] = []

    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = float(data.get("conf", ["-1"])[i]) if "conf" in data else -1.0
        if not txt:
            continue
        if conf >= 0:
            # фильтр по уверенности, но не слишком агрессивно
            if conf < 35:
                continue

        full_text_parts.append(txt)

        key = (data.get("block_num", [0])[i], data.get("par_num", [0])[i], data.get("line_num", [0])[i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        rec = blocks_map.get(key)
        if not rec:
            blocks_map[key] = {
                "text": [txt],
                "bbox": [x, y, w, h],
            }
        else:
            rec["text"].append(txt)
            bx, by, bw, bh = rec["bbox"]
            # union bbox
            x2 = max(bx + bw, x + w)
            y2 = max(by + bh, y + h)
            nx = min(bx, x)
            ny = min(by, y)
            rec["bbox"] = [nx, ny, x2 - nx, y2 - ny]

    blocks = []
    for rec in blocks_map.values():
        line = " ".join(rec["text"]).strip()
        if not line:
            continue
        x, y, w, h = rec["bbox"]
        # очень грубая догадка типа элемента по геометрии/форме/длине
        kind = "text"
        if len(line) <= 20 and h >= 28:
            kind = "title_or_button"
        if len(line) <= 14 and w <= 220 and h >= 30:
            kind = "button_like"
        blocks.append({"text": line, "bbox": [x, y, w, h], "kind_guess": kind})

    return {"ok": True, "text": " ".join(full_text_parts).strip(), "blocks": blocks}


# =============================
# LLM (hybrid: image + OCR)
# =============================
def analyze_ui_with_openai(image_b64: str, ocr: Dict[str, Any]) -> Dict[str, Any]:
    ocr_text = (ocr.get("text") or "").strip()
    blocks = ocr.get("blocks") or []

    # ограничим размер, чтобы не улететь по токенам
    blocks_short = blocks[:80]

    prompt = f"""
Ты — старший продуктовый дизайнер и требовательный дизайн-ревьюер.
Говоришь по-русски. Без мата. Без сюсюканья.
Если хорошо — хвали конкретно. Если плохо — ругай конкретно и предлагай улучшения.

Важно:
- Никаких технических деталей (пиксели, коды цветов, расчёты).
- Про шрифт/палитру — только предположения ("похоже на sans-serif типа Inter/SF/Roboto").
- Учитывай контекст. Не путай заголовки и кнопки. Не выдумывай то, чего нет.
- Для текста ориентируйся на OCR-данные ниже, но сверяй с картинкой.

OCR_STATUS: {"OK" if ocr.get("ok") else "FAIL"}
OCR_TEXT (может быть неполным):
{ocr_text[:1800]}

OCR_BLOCKS (список строк с bbox и грубым guess):
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
        return {
            "description": (out_text[:900] or "Модель вернула пустой ответ."),
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
        "Нажми кнопку снизу или просто отправь картинку.",
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
        f"pong ✅\nMODEL: <code>{html_escape(LLM_MODEL)}</code>\nOCR: <code>{'on' if OCR_AVAILABLE else 'off'}</code>",
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
        progress = await m.answer("⏳ Принял. Загружаю…", reply_markup=keyboard)
        await animate_progress(progress, title="🔍 Смотрю внимательно…")

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

        # OCR stage
        await progress_set(progress, "🧾 Читаю текст…", 3)
        ocr = ocr_extract(img)

        # если OCR провалился — сообщим коротко (без тех. деталей), но не остановим анализ
        if not ocr.get("ok"):
            # не спамим: просто добавим отдельное дружелюбное сообщение
            await m.answer(
                "⚠️ Текст на скрине читается плохо (или OCR недоступен). "
                "Я всё равно попробую разобрать по картинке, но точность может просесть.",
                reply_markup=keyboard,
            )

        await progress_set(progress, "🧠 Думаю…", 5)

        try:
            result = analyze_ui_with_openai(img_to_base64_png(img), ocr)
        except Exception:
            await m.answer(
                "⚠️ Я не смог нормально разобрать этот экран.\n\n"
                "Чаще всего это из-за:\n"
                "• слишком мелкого/размытого текста\n"
                "• сильной перегруженности экрана\n"
                "• обрезанного интерфейса\n\n"
                "Сделай так:\n"
                "— пришли скрин крупнее\n"
                "— обрежь лишнее вокруг\n"
                "— если это веб: зум 125–150% и пересними",
                reply_markup=keyboard,
            )
            return

        await progress_set(progress, "✅ Готово.", 6)

        desc = html_escape(result.get("description", "")) or "—"
        visual = html_escape(result.get("visual", "")) or "—"
        text = html_escape(result.get("text", "")) or "—"
        score = clamp_score(result.get("score", 6))

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
    print(f"✅ Design Review Partner starting… model={LLM_MODEL}, OCR_AVAILABLE={OCR_AVAILABLE}, CV_AVAILABLE={CV_AVAILABLE}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
