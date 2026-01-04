# bot.py
import os
import re
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, Tuple

import httpx
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup
from aiogram.client.default import DefaultBotProperties

try:
    from PIL import Image
except Exception:
    Image = None

# OCR optional
try:
    import pytesseract
except Exception:
    pytesseract = None


# ---------------------------
# Env / Config
# ---------------------------

def load_env():
    # ВАЖНО: не используем find_dotenv() (он у тебя падал AssertionError)
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)


def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


load_env()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in .env or environment")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LLM_ENABLED = env_bool("LLM_ENABLED", True)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

OCR_ENABLED = env_bool("OCR_ENABLED", True)
OCR_LANG = os.getenv("OCR_LANG", "rus+eng").strip()
OCR_MIN_CONF = env_int("OCR_MIN_CONF", 55)

RULES_PATH = os.getenv("RULES_PATH", "rules.json").strip()

# Telegram UI
BTN_SEND_SCREEN = "🖼 Закинуть скрин"
BTN_HELP = "ℹ️ Как пользоваться"
BTN_PING = "🏓 Ping"


# ---------------------------
# Rules loader (optional)
# ---------------------------

def load_rules_text() -> str:
    """
    Мы не заставляем LLM работать строго по JSON-структуре.
    Но даём краткое резюме правил, если rules.json есть.
    """
    p = Path(__file__).with_name(RULES_PATH)
    if not p.exists():
        return "Правила: (rules.json не найден; ревью делаем по общим принципам понятности и B2B-тона)."

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return "Правила: (rules.json есть, но не удалось прочитать JSON; ревью делаем по общим принципам)."

    # Ожидаем, что там есть список правил/категорий. Но не привязываемся.
    # Соберём в текст: самые важные принципы.
    chunks = []
    if isinstance(data, dict):
        if "principles" in data and isinstance(data["principles"], list):
            for x in data["principles"][:20]:
                if isinstance(x, str) and x.strip():
                    chunks.append(f"- {x.strip()}")
        if "rules" in data and isinstance(data["rules"], list):
            for r in data["rules"][:30]:
                if isinstance(r, dict):
                    t = r.get("title") or r.get("name") or r.get("id")
                    d = r.get("description") or r.get("what") or r.get("problem")
                    if t and d:
                        chunks.append(f"- {str(t).strip()}: {str(d).strip()}")
    if not chunks:
        return "Правила: (rules.json прочитан, но структура нестандартная; ревью делаем по общим принципам + здравому смыслу)."

    return "Коротко о правилах/принципах:\n" + "\n".join(chunks)


RULES_TEXT = load_rules_text()


# ---------------------------
# Helpers
# ---------------------------

def ascii_progress_frame(step: int, total: int = 10, label: str = "Обрабатываю") -> str:
    filled = max(0, min(total, step))
    bar = "#" * filled + "-" * (total - filled)
    # чуть больше ASCII-вайба
    return (
        f"{label}...\n"
        f"[{bar}] {filled}/{total}\n"
        f"╭──────────────╮\n"
        f"│   {('▰' * filled).ljust(total)}   │\n"
        f"╰──────────────╯"
    )


def clean_text(s: str) -> str:
    # чистим странные пробелы/мусор
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def score_clamp(x: int) -> int:
    return max(1, min(10, x))


def guess_font_family_from_image_text(ocr_text: str) -> str:
    # Мы честно "угадываем". Без статистик и размеров.
    # На деле шрифт по OCR почти не вытащить; делаем мягкий вывод.
    if not ocr_text:
        return "Не уверен (мало текста для угадывания)"
    # просто нейтральная формулировка
    return "Похоже на современный sans-serif (типа Inter / SF / Roboto) — без гарантий"


def ocr_extract_text(image_path: str) -> str:
    if not OCR_ENABLED:
        return ""
    if pytesseract is None or Image is None:
        return ""
    try:
        img = Image.open(image_path)
        # tesseract конфиг: табы/пробелы, нормальный режим
        data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
        words = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            conf = data.get("conf", [])[i]
            try:
                conf_i = int(float(conf))
            except Exception:
                conf_i = -1
            if txt and conf_i >= OCR_MIN_CONF:
                words.append(txt)
        return clean_text(" ".join(words))
    except Exception:
        return ""


async def call_openai_vision_review(
    image_bytes: bytes,
    ocr_text: str,
    rules_text: str,
    model: str,
) -> Tuple[str, str, str]:
    """
    Возвращает 3 текста: (что вижу), (визуал), (текст).
    Без JSON, чтобы не ловить "invalid JSON" и ошибки формата.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    # Жёстко задаём формат ответа: 3 секции простым текстом.
    system = (
        "Ты — придирчивый старший дизайн-ревьюер для B2B банка. "
        "Ты честный, иногда жёсткий, но без грубости и без мата. "
        "Если хорошо — похвали конкретно. Если плохо — ругай конкретно и предложи исправление. "
        "НЕ используй HTML-теги. Никаких JSON, никаких словарей вида {'key': ...}."
    )

    user = (
        "Задача: проанализируй скрин интерфейса.\n\n"
        "Выход: верни РОВНО три блока текста, в таком формате:\n"
        "1) WHAT_I_SEE: 2–6 предложений, что на экране происходит.\n"
        "2) VISUAL_REVIEW (SCORE X/10): 5–12 пунктов. Только про визуал/UX: иерархия, отступы, выравнивание, перегруз, контраст (без чисел), консистентность, читаемость. "
        "Про шрифт — ТОЛЬКО предположение о семействе (например 'sans-serif типа Inter/SF/Roboto'), без размеров, медиан и точных цветов.\n"
        "3) TEXT_REVIEW (SCORE Y/10): 6–14 пунктов. Каждый пункт: 'Проблема → Почему плохо → Как исправить'. "
        "Обязательно чтобы было понятно, что именно не так и что сделать.\n\n"
        "Контекст правил (суть):\n"
        f"{rules_text}\n\n"
        "Если OCR текст есть — используй его как подсказку, но приоритет у того, что видно на экране.\n"
        f"OCR_TEXT (может быть неполный): {ocr_text or '(нет)'}\n"
    )

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "max_output_tokens": 900,
    }

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        out = r.json()

    # вытаскиваем текст из Responses API
    text_parts = []
    for item in out.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text" and c.get("text"):
                text_parts.append(c["text"])
    full = clean_text("\n".join(text_parts))
    if not full:
        raise RuntimeError("LLM returned empty output")

    # парсим 3 секции по маркерам
    # (делаем устойчиво: даже если модель чуть отклонится, мы вытащим максимально)
    what = ""
    visual = ""
    text = ""

    # нормализуем
    norm = full.replace("\r\n", "\n")

    def extract_block(marker: str) -> str:
        m = re.search(rf"{marker}\s*:\s*", norm, flags=re.IGNORECASE)
        if not m:
            return ""
        start = m.end()
        # до следующего маркера или конец
        next_m = re.search(r"(WHAT_I_SEE\s*:|VISUAL_REVIEW\s*\(|TEXT_REVIEW\s*\()", norm[start:], flags=re.IGNORECASE)
        if next_m:
            return clean_text(norm[start:start + next_m.start()])
        return clean_text(norm[start:])

    what = extract_block("WHAT_I_SEE")
    # для VISUAL/TEXT удобнее вырезать по строкам
    # попробуем найти секции через заголовки
    vm = re.search(r"VISUAL_REVIEW\s*\(.*?\)\s*:", norm, flags=re.IGNORECASE)
    tm = re.search(r"TEXT_REVIEW\s*\(.*?\)\s*:", norm, flags=re.IGNORECASE)

    if vm:
        start = vm.end()
        end = tm.start() if tm else len(norm)
        visual = clean_text(norm[start:end])

    if tm:
        start = tm.end()
        text = clean_text(norm[start:])

    # fallback: если формат не соблюдён, просто разрежем аккуратно
    if not (what and visual and text):
        # попробуем грубо поделить на 3 части по пустым строкам
        parts = [p.strip() for p in re.split(r"\n\s*\n", norm) if p.strip()]
        if not what and parts:
            what = parts[0]
        if not visual and len(parts) >= 2:
            visual = parts[1]
        if not text and len(parts) >= 3:
            text = "\n\n".join(parts[2:])

    return what.strip(), visual.strip(), text.strip()


# ---------------------------
# Telegram bot setup
# ---------------------------

kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_SEND_SCREEN)],
        [KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_PING)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Кидай скрин — я разнесу (по делу).",
)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=None),  # без HTML, чтобы не ловить entity errors
)
dp = Dispatcher()


@dp.message(F.text.in_({"/start", "start"}))
async def cmd_start(m: Message):
    await m.answer(
        "Я — партнёр по дизайн-ревью.\n"
        "Кидай скрин интерфейса — я придирчиво разберу UI/UX и тексты.\n\n"
        "Жми «🖼 Закинуть скрин» или просто отправь картинку сюда.",
        reply_markup=kb,
    )


@dp.message(F.text == BTN_HELP)
async def cmd_help(m: Message):
    await m.answer(
        "Как пользоваться:\n"
        "1) Отправь скрин интерфейса.\n"
        "2) Я покажу прогресс ASCII.\n"
        "3) Потом пришлю 3 сообщения:\n"
        "   • что вижу на экране\n"
        "   • визуальный разбор + оценка\n"
        "   • разбор текста + оценка\n\n"
        "Подсказка: чем крупнее текст на скрине — тем точнее придирки.",
        reply_markup=kb,
    )


@dp.message(F.text == BTN_PING)
async def cmd_ping(m: Message):
    await m.answer(
        f"pong ✅\n"
        f"LLM_ENABLED={LLM_ENABLED}\n"
        f"MODEL={LLM_MODEL}\n"
        f"OCR_ENABLED={OCR_ENABLED} ({OCR_LANG})\n"
        f"RULES={RULES_PATH}",
        reply_markup=kb,
    )


@dp.message(F.text == BTN_SEND_SCREEN)
async def ask_screen(m: Message):
    await m.answer("Ок. Кидай скриншот. Я посмотрю и докопаюсь по делу 🙂", reply_markup=kb)


@dp.message(F.photo)
async def handle_photo(m: Message):
    # скачиваем фото (берём самое большое)
    photo = m.photo[-1]
    file = await bot.get_file(photo.file_id)
    # временный путь
    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_path = tmp_dir / f"tg_{photo.file_unique_id}.png"
    await bot.download_file(file.file_path, destination=str(img_path))

    # Прогресс-бар ASCII (редактируем одно сообщение)
    progress_msg = await m.answer(ascii_progress_frame(1, label="Загружаю"), reply_markup=kb)

    # читаем байты
    image_bytes = img_path.read_bytes()

    # OCR (опционально)
    await progress_msg.edit_text(ascii_progress_frame(3, label="Достаю текст (OCR)"))
    ocr_text = ocr_extract_text(str(img_path))

    # Угадаем “семейство шрифта” очень мягко (и честно)
    font_guess = guess_font_family_from_image_text(ocr_text)

    # LLM review
    if not LLM_ENABLED:
        await progress_msg.edit_text(ascii_progress_frame(10, label="Готово"))
        await m.answer(
            "LLM выключен (LLM_ENABLED=false).\n"
            "Я сейчас могу сделать только OCR-сводку.\n\n"
            f"Текст (OCR): {ocr_text or '(не извлёк)'}\n"
            f"Шрифт (предположение): {font_guess}",
            reply_markup=kb,
        )
        return

    await progress_msg.edit_text(ascii_progress_frame(5, label="Думаю (LLM)"))

    try:
        what, visual, text = await call_openai_vision_review(
            image_bytes=image_bytes,
            ocr_text=ocr_text,
            rules_text=RULES_TEXT,
            model=LLM_MODEL,
        )
    except httpx.HTTPStatusError as e:
        await progress_msg.edit_text(ascii_progress_frame(10, label="Упс"))
        await m.answer(
            f"⚠️ Ошибка запроса к LLM: {e.response.status_code}\n"
            f"{clean_text(e.response.text)[:1200]}",
            reply_markup=kb,
        )
        return
    except Exception as e:
        await progress_msg.edit_text(ascii_progress_frame(10, label="Упс"))
        await m.answer(f"⚠️ Упало при обработке: {e}", reply_markup=kb)
        return

    await progress_msg.edit_text(ascii_progress_frame(10, label="Готово"))

    # 1) что вижу
    await m.answer(
        "👀 Что я вижу на скрине:\n"
        f"{what}\n\n"
        f"Шрифт (предположение): {font_guess}",
        reply_markup=kb,
    )

    # 2) визуал
    await m.answer(
        "🎛 Визуальный разбор:\n"
        f"{visual}",
        reply_markup=kb,
    )

    # 3) текст
    await m.answer(
        "✍️ Разбор текста:\n"
        f"{text}",
        reply_markup=kb,
    )


@dp.message()
async def fallback(m: Message):
    await m.answer(
        "Я понимаю либо команды, либо картинку.\n"
        "Кидай скриншот интерфейса — и я устрою ревью.",
        reply_markup=kb,
    )


async def main():
    print(f"✅ Bot starting… LLM_ENABLED={LLM_ENABLED}, model={LLM_MODEL}, OCR_ENABLED={OCR_ENABLED}, rules={RULES_PATH}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
