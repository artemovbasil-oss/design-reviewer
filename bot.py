# bot.py (aiogram 3.7.0) — Design Review Partner (Railway-safe)
# FIXES:
# 1) Больше НЕТ спама ASCII-сообщениями: если Telegram запретил edit — анимация молча останавливается.
# 2) Ошибки анализа — по-человечески: почему могло упасть и что сделать.
# 3) Лок на чат: один скрин за раз, чтобы прогресс/ответы не путались.
# 4) 3 сообщения: что вижу / визуал (оценка) / тексты.

import os
import re
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest

from openai import OpenAI


# =============================
# Optional local .env loader (NO python-dotenv dependency)
# =============================
def load_local_env_file() -> None:
    """
    Railway: не нужен.
    Локально: если рядом есть .env — загрузим простым парсером.
    Формат: KEY=VALUE (без export)
    """
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
    input_field_placeholder="Кидай скрин — я разберу без сантиментов (но по делу).",
)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()


# =============================
# Concurrency: per-chat lock
# =============================
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
    """
    Пытаемся отредактировать сообщение.
    Если Telegram запрещает edit — НЕ шлём новое (чтобы не спамить).
    Возвращаем True/False (получилось ли отредактировать).
    """
    try:
        await msg.edit_text(text)
        return True
    except TelegramBadRequest:
        return False


async def animate_progress(msg: Message, title: str = "🔍 Смотрю внимательно…") -> None:
    """
    ASCII-анимация прогресса.
    Если edit запрещён — молча прекращаем, не спамим новыми сообщениями.
    """
    last_text = None
    for i in range(6):
        cur_text = f"{title} {spinner_frame(i)}\n<code>{ascii_frame(i)}</code>"
        ok = await safe_edit_text(msg, cur_text)
        # если нельзя редактировать — выходим
        if not ok:
            break
        # защита от бессмысленных правок (иногда Telegram "не любит" частые одинаковые)
        if last_text == cur_text:
            break
        last_text = cur_text
        await asyncio.sleep(0.22)


def parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def analyze_ui_with_openai(image_b64: str) -> Dict[str, Any]:
    """
    Returns dict:
      description, score, visual, text
    """
    prompt = """
Ты — старший продуктовый дизайнер и требовательный дизайн-ревьюер.
Говоришь по-русски. Без мата. Без сюсюканья.
Если хорошо — хвали конкретно. Если плохо — ругай конкретно и предлагай улучшения.

Важно:
- Никаких технических деталей (пиксели, коды цветов, расчёты).
- Про шрифт/палитру — только предположения (например: "похоже на sans-serif типа Inter/SF/Roboto").
- Учитывай контекст: заголовок ≠ кнопка. Не выдумывай элементы, которых нет.

Верни СТРОГО JSON:
{
  "description": "2–6 предложений: что происходит на экране",
  "score": 1-10,
  "visual": "5–12 пунктов: визуал/UX (с похвалой, если есть)",
  "text": "6–14 пунктов: текст (каждый пункт: Проблема → Почему плохо → Как исправить)"
}
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
        max_output_tokens=900,
    )

    out_text = ""
    for item in getattr(resp, "output", []) or []:
        for c in item.content or []:
            if getattr(c, "type", None) == "output_text":
                out_text += getattr(c, "text", "") + "\n"

    out_text = out_text.strip()
    data = parse_llm_json(out_text)
    if not data:
        # fallback: хоть что-то
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


async def progress_set(msg: Message, title: str, i: int) -> None:
    """
    Единичная установка прогресса. Если edit запрещён — просто молчим.
    """
    await safe_edit_text(msg, f"{title} {spinner_frame(i)}\n<code>{ascii_frame(i)}</code>")


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
        f"pong ✅\nMODEL: <code>{html_escape(LLM_MODEL)}</code>",
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

        # Попробуем анимировать (если Telegram запретит edit — просто остановится)
        await animate_progress(progress, title="🔍 Смотрю внимательно…")

        photo = m.photo[-1]
        file = await bot.get_file(photo.file_id)

        bio = BytesIO()
        await bot.download_file(file.file_path, destination=bio)
        bio.seek(0)

        try:
            img = Image.open(bio).convert("RGBA")
        except Exception:
            # Даже если edit запрещён — отправим отдельным сообщением, чтобы пользователь увидел.
            await m.answer("⚠️ Не смог открыть картинку. Пришли другой файл.", reply_markup=keyboard)
            return

        await progress_set(progress, "🧠 Думаю…", 5)

        try:
            result = analyze_ui_with_openai(img_to_base64_png(img))
        except Exception:
            # Нормальное человекочитаемое объяснение
            await m.answer(
                "⚠️ Я не смог нормально разобрать этот экран.\n\n"
                "Обычно это бывает, если:\n"
                "• текст слишком мелкий или размытый\n"
                "• скрин перегружен деталями\n"
                "• интерфейс обрезан или снят с блюром\n\n"
                "Что сделать:\n"
                "— пришли скрин крупнее\n"
                "— обрежь лишнее вокруг экрана\n"
                "— если это веб — сделай зум 125–150% и пересними",
                reply_markup=keyboard,
            )
            return

        await progress_set(progress, "✅ Готово.", 6)

        desc = html_escape(result.get("description", "")) or "—"
        visual = html_escape(result.get("visual", "")) or "—"
        text = html_escape(result.get("text", "")) or "—"
        score = clamp_score(result.get("score", 6))

        # 3 сообщения отчёта
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


# =============================
# Run
# =============================
async def main():
    print(f"✅ Design Review Partner starting… model={LLM_MODEL}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
