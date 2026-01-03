# bot.py
import os, io, json, base64, asyncio, warnings
from typing import List, Dict, Any, Optional

import requests
from PIL import Image
import pytesseract
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums.parse_mode import ParseMode
from aiogram.client.default import DefaultBotProperties

from html import escape as htmlesc

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# -------------------------------------------------------
# env
# -------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

OCR_LANG = os.getenv("OCR_LANG", "rus+eng")
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "55"))
OCR_MIN_WORD_CONF = float(os.getenv("OCR_MIN_WORD_CONF", "45"))

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in .env or environment")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env or environment")

# -------------------------------------------------------
# aiogram setup
# -------------------------------------------------------
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🖼 Закинуть скрин и получить ревью")],
        [KeyboardButton(text="ℹ️ Что умеешь?")],
    ],
    resize_keyboard=True,
    input_field_placeholder="Кидай скрин 👇",
)

# -------------------------------------------------------
# helpers
# -------------------------------------------------------
def ascii_bar(p: int, width: int = 22) -> str:
    p = max(0, min(100, p))
    filled = int(round(width * p / 100))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {p}%"

SPINNER = ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]
PULSE = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"]

async def progress_edit(msg: Message, title: str, p: int, note: str = "", tick: int = 0):
    spin = SPINNER[tick % len(SPINNER)]
    pulse = PULSE[tick % len(PULSE)]
    text = (
        f"<b>{htmlesc(title)}</b> {spin}\n"
        f"<code>{htmlesc(ascii_bar(p))}</code>\n"
        f"<code>processing {pulse}{pulse}{pulse}</code>"
    )
    if note:
        text += f"\n{htmlesc(note)}"
    await msg.edit_text(text)

async def animate_stage(
    msg: Message,
    title: str,
    p_from: int,
    p_to: int,
    note: str,
    steps: int = 7,
    delay: float = 0.08,
):
    for i in range(steps):
        p = int(p_from + (p_to - p_from) * (i / max(1, steps - 1)))
        await progress_edit(msg, title, p, note, tick=i)
        await asyncio.sleep(delay)

def image_to_data_url_png(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, dict):
        vals = list(x.values())
        return vals if vals else list(x.keys())
    if isinstance(x, str):
        return [x]
    return [str(x)]

def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("description", "text", "value"):
            if k in x and isinstance(x[k], str):
                return x[k].strip()
        return " ".join(str(v) for v in x.values()).strip()
    return str(x).strip()

# -------------------------------------------------------
# OCR
# -------------------------------------------------------
def ocr_lines(pil_img: Image.Image) -> List[Dict[str, Any]]:
    gray = pil_img.convert("L")
    data = pytesseract.image_to_data(
        gray,
        lang=OCR_LANG,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < OCR_MIN_WORD_CONF:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append({"text": txt, "bbox": (x, y, w, h), "conf": conf})

    words.sort(key=lambda w: (w["bbox"][1], w["bbox"][0]))

    lines: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    last_y: Optional[int] = None
    y_thresh = 12

    for w in words:
        y = w["bbox"][1]
        if last_y is None or abs(y - last_y) <= y_thresh:
            cur.append(w)
            if last_y is None:
                last_y = y
        else:
            if cur:
                lines.append(cur)
            cur = [w]
            last_y = y
    if cur:
        lines.append(cur)

    result = []
    for ln in lines:
        text = " ".join(w["text"] for w in ln).strip()
        if not text:
            continue
        conf = sum(w["conf"] for w in ln) / max(1, len(ln))
        if conf < OCR_MIN_CONF:
            continue

        xs = [w["bbox"][0] for w in ln]
        ys = [w["bbox"][1] for w in ln]
        ws = [w["bbox"][2] for w in ln]
        hs = [w["bbox"][3] for w in ln]
        x0, y0 = min(xs), min(ys)
        x1 = max(xs[i] + ws[i] for i in range(len(xs)))
        y1 = max(ys[i] + hs[i] for i in range(len(ys)))
        result.append(
            {
                "text": text,
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "conf": conf,
            }
        )
    return result

# -------------------------------------------------------
# OpenAI Responses API (Structured Outputs)
# -------------------------------------------------------
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

def extract_responses_text(data: Dict[str, Any]) -> str:
    t = data.get("output_text")
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = data.get("output", [])
    chunks: List[str] = []
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for c in item.get("content", []) or []:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") in ("output_text", "text"):
                        txt = c.get("text")
                        if isinstance(txt, str) and txt:
                            chunks.append(txt)
    return "".join(chunks).strip()

def openai_design_review(image_png_bytes: bytes, ocr_snippet: str) -> Dict[str, Any]:
    img_url = image_to_data_url_png(image_png_bytes)

    system = (
        "Ты — старший дизайнер (design lead) и партнёр по дизайн-ревью для банковского B2B продукта.\n"
        "Тон: дружелюбно, но честно. Если плохо — говоришь прямо (без мата). Если хорошо — хвалишь конкретно.\n"
        "Никаких технических деталей типа размеров/медиан/пикселей/HEX-цветов.\n"
        "Про шрифты: можно только предположить семейство (например: Inter / SF Pro / Roboto / Helvetica / PT Sans) и сказать, что это гипотеза.\n"
        "Структура каждого замечания: что хорошо (если есть) → что плохо → почему → что сделать.\n"
        "Верни строго валидный JSON по схеме."
    )

    user = (
        "Сделай дизайн-ревью скриншота.\n"
        "Нужно 3 части:\n"
        "1) what_i_see: кратко опиши экран человеческим языком\n"
        "2) visual_report: минимум 5 пунктов\n"
        "3) text_report: минимум 5 пунктов\n"
        "Поставь оценки ui/ux/copy/overall (1–10) и вердикт: что чинить первым.\n\n"
        "OCR (может шуметь):\n"
        f"{ocr_snippet}\n"
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "what_i_see": {"type": "string"},
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ui": {"type": "integer", "minimum": 1, "maximum": 10},
                    "ux": {"type": "integer", "minimum": 1, "maximum": 10},
                    "copy": {"type": "integer", "minimum": 1, "maximum": 10},
                    "overall": {"type": "integer", "minimum": 1, "maximum": 10},
                    "verdict": {"type": "string"},
                },
                "required": ["ui", "ux", "copy", "overall", "verdict"],
            },
            "visual_report": {
                "type": "array",
                "minItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "good": {"type": "string"},
                        "issue": {"type": "string"},
                        "why": {"type": "string"},
                        "fix": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "med", "low"]},
                    },
                    "required": ["good", "issue", "why", "fix", "priority"],
                },
            },
            "text_report": {
                "type": "array",
                "minItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "good": {"type": "string"},
                        "issue": {"type": "string"},
                        "why": {"type": "string"},
                        "fix": {"type": "string"},
                        "example": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "med", "low"]},
                    },
                    "required": ["good", "issue", "why", "fix", "example", "priority"],
                },
            },
            "fonts_guess": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["what_i_see", "scores", "visual_report", "text_report", "fonts_guess"],
    }

    payload = {
        "model": LLM_MODEL,
        "instructions": system,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": img_url},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "design_review_schema_v2",
                "schema": schema,
            }
        },
        "max_output_tokens": 1500,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=150)
    r.raise_for_status()
    data = r.json()

    text = extract_responses_text(data)
    if not text:
        raise RuntimeError("Empty response text from OpenAI")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise

# -------------------------------------------------------
# formatting
# -------------------------------------------------------
def fmt_scores(scores: Dict[str, Any]) -> str:
    def clamp10(x):
        try:
            v = int(x)
        except Exception:
            return None
        return max(1, min(10, v))

    ui = clamp10(scores.get("ui"))
    ux = clamp10(scores.get("ux"))
    cp = clamp10(scores.get("copy"))
    ov = clamp10(scores.get("overall"))
    verdict = safe_text(scores.get("verdict")) or "Начни с основы: ясная иерархия + понятные действия."

    def bar10(v: Optional[int]) -> str:
        if v is None:
            return "??????????"
        return "■" * v + "□" * (10 - v)

    def s(v: Optional[int]) -> str:
        return "?" if v is None else str(v)

    return (
        "<b>Оценка (1–10)</b>\n"
        f"UI   {s(ui)}/10  <code>{bar10(ui)}</code>\n"
        f"UX   {s(ux)}/10  <code>{bar10(ux)}</code>\n"
        f"Copy {s(cp)}/10  <code>{bar10(cp)}</code>\n"
        f"Итог {s(ov)}/10  <code>{bar10(ov)}</code>\n"
        f"<b>Вердикт:</b> {htmlesc(verdict)}"
    )

def format_visual_report(items: List[Dict[str, Any]]) -> str:
    pr_map = {"high": "🔴", "med": "🟠", "low": "🟡"}
    if not items:
        return "<b>2) Визуальная часть</b>\nНечего обсуждать: или скрин без интерфейса, или слишком мелко."
    lines = ["<b>2) Визуальная часть</b>"]
    for it in items[:12]:
        pr = pr_map.get(str(it.get("priority", "med")).lower(), "🟠")
        good = htmlesc(safe_text(it.get("good")))
        issue = htmlesc(safe_text(it.get("issue")))
        why = htmlesc(safe_text(it.get("why")))
        fix = htmlesc(safe_text(it.get("fix")))

        # чуть мягче, по-дружески
        block = f"{pr} <b>{issue}</b>"
        if good:
            block += f"\n— 👍 что хорошо: {good}"
        block += f"\n— почему это важно: {why}\n— как улучшить: {fix}"
        lines.append(block)
    return "\n\n".join(lines)

def format_text_report(items: List[Dict[str, Any]]) -> str:
    pr_map = {"high": "🔴", "med": "🟠", "low": "🟡"}
    if not items:
        return "<b>3) Текст</b>\nТекста не видно. Это уже проблема: человек не понимает, что происходит."
    lines = ["<b>3) Текст</b>"]
    for it in items[:12]:
        pr = pr_map.get(str(it.get("priority", "med")).lower(), "🟠")
        good = htmlesc(safe_text(it.get("good")))
        issue = htmlesc(safe_text(it.get("issue")))
        why = htmlesc(safe_text(it.get("why")))
        fix = htmlesc(safe_text(it.get("fix")))
        ex = htmlesc(safe_text(it.get("example")))

        block = f"{pr} <b>{issue}</b>"
        if good:
            block += f"\n— 👍 что хорошо: {good}"
        block += f"\n— почему это важно: {why}\n— как улучшить: {fix}"
        if ex:
            block += f"\n— пример: «{ex}»"
        lines.append(block)
    return "\n\n".join(lines)

# -------------------------------------------------------
# handlers
# -------------------------------------------------------
@dp.message(CommandStart())
async def start(m: Message):
    await m.answer(
        "👋 Привет. Я — <b>Design Review Partner</b>.\n\n"
        "Кидай скрин — я:\n"
        "• расскажу, что вижу\n"
        "• отмечу, что сделано хорошо\n"
        "• честно скажу, что мешает UX\n"
        "• предложу, как улучшить\n"
        "• поставлю оценку 1–10\n\n"
        "Давай, присылай скриншот.",
        reply_markup=main_kb,
    )

@dp.message(F.text == "ℹ️ Что умеешь?")
async def about(m: Message):
    await m.answer(
        "<b>Как я работаю</b>\n"
        "1) Ты кидаешь скрин\n"
        "2) Я показываю прогресс ASCII-анимацией\n"
        "3) Отправляю 4 сообщения:\n"
        "   • что вижу\n"
        "   • оценка и вердикт\n"
        "   • визуал\n"
        "   • текст\n\n"
        "Совет: если текст мелкий — пришли скрин покрупнее."
    )

@dp.message(F.text == "🖼 Закинуть скрин и получить ревью")
async def ask_screen(m: Message):
    await m.answer("Пришли скриншот (png/jpg).")

@dp.message(F.photo | F.document[(F.document.mime_type.startswith("image/"))])
async def handle_image(m: Message):
    if m.photo:
        file_id = m.photo[-1].file_id
    else:
        file_id = m.document.file_id

    file = await bot.get_file(file_id)
    f = await bot.download_file(file.file_path)
    raw = f.read()

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    prog = await m.answer("<b>Обрабатываю…</b>\n<code>[░░░░░░░░░░░░░░░░░░░░░░] 0%</code>")

    try:
        await animate_stage(prog, "Загрузка", 3, 18, "Принял скрин. Сейчас посмотрим.", steps=8)
        await animate_stage(prog, "OCR", 20, 45, "Вытаскиваю текст (где получится).", steps=9)

        lines = ocr_lines(img)
        ocr_texts = [ln["text"] for ln in lines[:30]]
        ocr_snippet = "\n".join(f"- {t}" for t in ocr_texts) if ocr_texts else "(текст почти не читается / мало текста)"

        await animate_stage(prog, "Ревью", 48, 88, "Собираю мысли. Буду честным.", steps=12)

        review = await asyncio.to_thread(openai_design_review, raw, ocr_snippet)

        await animate_stage(prog, "Финал", 90, 99, "Оформляю отчёт.", steps=7)

        what = safe_text(review.get("what_i_see")) or "Не смог уверенно понять, что на экране."
        scores = review.get("scores") if isinstance(review.get("scores"), dict) else {}
        visual_items = review.get("visual_report") if isinstance(review.get("visual_report"), list) else []
        text_items = review.get("text_report") if isinstance(review.get("text_report"), list) else []
        fonts_guess = safe_list(review.get("fonts_guess"))

        msg1 = "<b>1) Что я вижу на скриншоте</b>\n" + htmlesc(what)
        if fonts_guess:
            msg1 += "\n\n<b>Шрифт (гипотеза)</b>: " + htmlesc(", ".join(map(str, fonts_guess[:4])))

        msg_scores = fmt_scores(scores)
        msg2 = format_visual_report(visual_items)
        msg3 = format_text_report(text_items)

        await progress_edit(prog, "Готово", 100, "Отправляю. Без сахара, но по делу.", tick=13)

        await m.answer(msg1)
        await m.answer(msg_scores)
        await m.answer(msg2)
        await m.answer(msg3)

    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:900]
        except Exception:
            pass
        await prog.edit_text(f"⚠️ Ошибка запроса к LLM: {htmlesc(str(e))}\n<code>{htmlesc(body)}</code>")
    except Exception as e:
        await prog.edit_text(f"⚠️ Упало при обработке: {htmlesc(str(e))}")

# -------------------------------------------------------
# run
# -------------------------------------------------------
if __name__ == "__main__":
    print(f"✅ Design Review Partner starting… OCR_LANG={OCR_LANG}, model={LLM_MODEL}")
    asyncio.run(dp.start_polling(bot))