import os
import asyncio
import base64
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup
from aiogram.enums import ParseMode

from openai import OpenAI

# ================== ENV ==================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# ================== INIT ==================
bot = Bot(
    token=BOT_TOKEN,
    default={"parse_mode": ParseMode.HTML},
)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

# ================== UI ==================
keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="🖼 Закинуть скриншот")]],
    resize_keyboard=True,
)

# ================== HELPERS ==================
def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

async def ascii_progress(msg: Message):
    frames = [
        "▱▱▱▱▱",
        "▰▱▱▱▱",
        "▰▰▱▱▱",
        "▰▰▰▱▱",
        "▰▰▰▰▱",
        "▰▰▰▰▰",
    ]
    for f in frames:
        await msg.edit_text(f"🔍 Анализирую интерфейс…\n`{f}`")
        await asyncio.sleep(0.3)

# ================== LLM ==================
def analyze_ui(image_b64: str) -> dict:
    prompt = """
Ты — старший продуктовый дизайнер.
Ты честный, придирчивый, но справедливый.

Задача:
1. Опиши, что ты видишь на интерфейсе
2. Дай общую оценку UI/UX по шкале 1–10
3. Напиши отчёт по визуалу (что плохо и как улучшить, если есть — похвали)
4. Напиши отчёт по текстам (ясность, тон, UX-копирайтинг)

Требования:
- Без технических деталей
- Без размеров, кодов цветов, пикселей
- Шрифты и стиль — только предположения
- Без мата, но строго
- Говори как опытный коллега

Ответ верни СТРОГО в JSON:
{
  "description": "...",
  "score": 0,
  "visual": "...",
  "text": "..."
}
"""

    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_base64": image_b64,
                    },
                ],
            }
        ],
    )

    return response.output_parsed[0]["content"][0]["json"]

# ================== HANDLERS ==================
@dp.message(F.text == "/start")
async def start(m: Message):
    await m.answer(
        "👋 Я твой дизайн-партнёр.\n\n"
        "Кидай скриншот интерфейса — я разберу его как на настоящем дизайн-ревью.\n"
        "Похвалю, если есть за что. Докопаюсь, если есть косяки.",
        reply_markup=keyboard,
    )

@dp.message(F.photo)
async def handle_image(m: Message):
    progress = await m.answer("⏳ Загружаю…")
    await ascii_progress(progress)

    photo = m.photo[-1]
    file = await bot.download(photo.file_id)
    image = Image.open(file)

    image_b64 = image_to_base64(image)

    try:
        result = analyze_ui(image_b64)
    except Exception as e:
        await progress.edit_text("⚠️ Ошибка анализа. Попробуй другой скрин.")
        raise e

    await progress.delete()

    await m.answer(f"👀 <b>Что я вижу</b>\n{result['description']}")
    await m.answer(f"📊 <b>Оценка</b>: {result['score']} / 10")
    await m.answer(f"🎨 <b>Визуал</b>\n{result['visual']}")
    await m.answer(f"✍️ <b>Тексты</b>\n{result['text']}")

# ================== RUN ==================
if __name__ == "__main__":
    print("✅ Design Review Partner is running")
    asyncio.run(dp.start_polling(bot))
