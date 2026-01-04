# bot.py — Design Review Partner (aiogram 3.7.0)
# Features:
# - Review screenshots (Telegram photo)
# - Review Figma frame links:
#   - Public files: uses Figma oEmbed thumbnail (no token required)
#   - With FIGMA_TOKEN: uses Figma Images API for higher quality

import os
import re
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs, quote_plus

import aiohttp
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

FIGMA_TOKEN = os.getenv("FIGMA_TOKEN", "").strip()  # optional
FIGMA_SCALE = float((os.getenv("FIGMA_SCALE", "2") or "2").strip())

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set (Railway Variables or local .env)")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set (Railway Variables or local .env)")

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# Telegram UI
# =============================
BTN_REVIEW = "◼︎ Закинуть на ревью (скрин/ссылка)"
BTN_HOW = "◻︎ Как это работает?"
BTN_PING = "▶︎ Ping"

keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_REVIEW)],
        [KeyboardButton(text=BTN_HOW), KeyboardButton(text=BTN_PING)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Кидай скрин или ссылку на Figma фрейм.",
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
        (text or "")
        .replace("&", "&amp;")
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
    raw = (raw or "").strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def strip_listish_wrappers(s: str) -> str:
    s = (s or "").strip()
    # common: "['a', 'b']" or '["a","b"]'
    if (s.startswith("[") and s.endswith("]")) and (("'" in s) or ('"' in s)):
        candidate = s
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
    if isinstance(x, list):
        items = []
        for it in x:
            t = str(it).strip()
            if t:
                items.append(f"{bullet}{t}")
        return "\n".join(items) if items else "—"
    if isinstance(x, dict):
        return "\n".join([f"{bullet}{k}: {v}" for k, v in x.items()]) or "—"
    s = strip_listish_wrappers(str(x or "").strip())
    return s or "—"


# =============================
# Retro ASCII progress
# =============================
def retro_bar(step: int, total: int = 12) -> str:
    step = max(0, min(step, total))
    return f"[{'#' * step}{'.' * (total - step)}]"


def retro_spinner(i: int) -> str:
    return ["|", "/", "-", "\\"][i % 4]


def retro_screen(title: str, i: int, step: int) -> str:
    bar = retro_bar(step)
    spin = retro_spinner(i)
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
    for i in range(12):
        current = await safe_edit_text_or_recreate(current, retro_screen(title, i, step=min(i, 10)))
        await asyncio.sleep(0.16)
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
# Figma link -> image bytes
# =============================
FIGMA_RE = re.compile(r"https?://(www\.)?figma\.com/(file|design)/([a-zA-Z0-9]+)")

def parse_figma_link(url: str) -> Optional[Dict[str, str]]:
    url = (url or "").strip()
    m = FIGMA_RE.search(url)
    if not m:
        return None
    file_key = m.group(3)

    try:
        p = urlparse(url)
        q = parse_qs(p.query)
        node_id = (q.get("node-id") or q.get("node_id") or [""])[0].strip()
    except Exception:
        node_id = ""

    node_id_api = node_id.replace("-", ":") if node_id else ""
    return {"file_key": file_key, "node_id": node_id, "node_id_api": node_id_api, "url": url}


async def figma_public_thumbnail(url: str) -> bytes:
    """
    For PUBLIC Figma links: use oEmbed to get thumbnail_url (works without token).
    Quality is limited but enough for review.
    """
    oembed = f"https://www.figma.com/api/oembed?url={quote_plus(url)}"
    async with aiohttp.ClientSession() as session:
        async with session.get(oembed, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"Figma oEmbed error: {r.status} {txt[:200]}")
            data = await r.json()

        thumb = data.get("thumbnail_url") or data.get("thumbnail_url_with_play_button")
        if not thumb:
            raise RuntimeError("Figma oEmbed did not return thumbnail_url (maybe not public?)")

        async with session.get(thumb, timeout=aiohttp.ClientTimeout(total=40)) as r2:
            if r2.status != 200:
                raise RuntimeError(f"Thumbnail download error: {r2.status}")
            return await r2.read()


async def figma_images_api_png(file_key: str, node_id_api: str, scale: float = 2.0) -> bytes:
    """
    Higher quality (requires FIGMA_TOKEN).
    """
    if not FIGMA_TOKEN:
        raise RuntimeError("FIGMA_TOKEN is not set")
    if not node_id_api:
        raise RuntimeError("No node-id in link")

    headers = {"X-Figma-Token": FIGMA_TOKEN}
    api_url = f"https://api.figma.com/v1/images/{file_key}"
    params = {
        "ids": node_id_api,
        "format": "png",
        "scale": str(scale),
        "use_absolute_bounds": "true",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(api_url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"Figma images API error: {r.status} {txt[:200]}")
            data = await r.json()

        images = (data.get("images") or {})
        img_url = images.get(node_id_api)
        if not img_url:
            err = data.get("err") or "No image URL returned (wrong node-id or no access?)"
            raise RuntimeError(f"Figma: {err}")

        async with session.get(img_url, timeout=aiohttp.ClientTimeout(total=60)) as r2:
            if r2.status != 200:
                raise RuntimeError(f"Figma image download error: {r2.status}")
            return await r2.read()


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
            max_output_tokens=650,
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
    for b in blocks[:90]:
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
            max_output_tokens=900,
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
        # fallback: show raw (short)
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
# Core review runner
# =============================
async def run_review_for_image(m: Message, img: Image.Image, title: str = "SCAN") -> None:
    # upscale small images
    w, h = img.size
    if max(w, h) < 1400:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)

    img_b64 = img_to_base64_png(img)

    progress = await m.answer("SCAN\n<pre>[............]</pre>")
    progress = await animate_progress(progress, title=title)

    progress = await set_progress(progress, "OCR", i=1, step=4)

    extracted = {"ok": False, "text": "", "blocks": []}
    ocr = ocr_extract(img)
    if ocr.get("ok") and len((ocr.get("text") or "").strip()) >= 12:
        extracted = {"ok": True, "text": ocr.get("text", ""), "blocks": ocr.get("blocks", [])}
    else:
        extracted = llm_extract_text_structure(img_b64)
        if not extracted.get("ok"):
            extracted = {"ok": False, "text": ocr.get("text", ""), "blocks": ocr.get("blocks", [])}

    progress = await set_progress(progress, "REVIEW", i=2, step=8)
    result = analyze_ui_with_openai(img_b64, extracted)
    progress = await set_progress(progress, "DONE", i=3, step=12)

    desc = html_escape(str(result.get("description", "")).strip()) or "—"
    score = clamp_score(result.get("score", 6))

    visual_txt = html_escape(bullets_from_any(result.get("visual", "—"), bullet="• "))
    text_txt = html_escape(bullets_from_any(result.get("text", "—"), bullet="• "))

    await m.answer(f"<b>Что вижу</b>\n{desc}", reply_markup=keyboard)
    await m.answer(f"<b>Визуал</b> — оценка: <b>{score}/10</b>\n{visual_txt}", reply_markup=keyboard)
    await m.answer(f"<b>Тексты</b>\n{text_txt}", reply_markup=keyboard)


async def run_review_for_figma_link(m: Message, url: str) -> None:
    info = parse_figma_link(url)
    if not info:
        await m.answer("Не вижу нормальную ссылку Figma. Пришли URL на фрейм/экран.", reply_markup=keyboard)
        return

    # Prefer Images API if token+node-id exist (higher quality)
    png_bytes: Optional[bytes] = None
    used = "FIGMA"

    try:
        if FIGMA_TOKEN and info.get("node_id_api"):
            png_bytes = await figma_images_api_png(
                file_key=info["file_key"],
                node_id_api=info["node_id_api"],
                scale=FIGMA_SCALE,
            )
        else:
            # Public-only mode: oEmbed thumbnail
            png_bytes = await figma_public_thumbnail(info["url"])
    except Exception as e:
        await m.answer(
            "Не смог вытащить картинку из Figma.\n"
            "Условия:\n"
            "• ссылка должна быть на публичный файл\n"
            "• (или добавь FIGMA_TOKEN для приватных)\n\n"
            f"Причина: <code>{html_escape(str(e))}</code>",
            reply_markup=keyboard,
        )
        return

    try:
        img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        await m.answer("Скачалось что-то странное — не открылось как PNG/JPG.", reply_markup=keyboard)
        return

    await run_review_for_image(m, img, title=used)


# =============================
# Handlers
# =============================
@dp.message(F.text.in_({"/start", "start"}))
async def start(m: Message):
    await m.answer(
        "<b>Design Review Partner</b>\n\n"
        "Я принимаю на ревью:\n"
        "• скриншоты интерфейса (картинки)\n"
        "• ссылки на фреймы Figma — <b>если файл публичный</b>\n\n"
        "Кидай — разберу. Если хорошо, похвалю конкретно. Если плохо — докопаюсь и предложу, как чинить.",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_REVIEW)
async def ask_review(m: Message):
    await m.answer(
        "Ок. Отправь:\n"
        "1) скриншот (картинку) <b>или</b>\n"
        "2) ссылку на Figma фрейм (публичный файл).\n\n"
        "Жду.",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_HOW)
async def how_it_works(m: Message):
    await m.answer(
        "<b>Как это работает</b>\n"
        "1) Закидываешь скрин или ссылку на Figma-фрейм (публичный).\n"
        "2) Я показываю ретро-прогресс.\n"
        "3) Потом 3 сообщения: что вижу / визуал / тексты.\n\n"
        "Если текст мелкий — пришли скрин крупнее или обрежь лишнее.",
        reply_markup=keyboard,
    )


@dp.message(F.text == BTN_PING)
async def ping(m: Message):
    figma_mode = "public-only" if not FIGMA_TOKEN else "token"
    await m.answer(
        "pong\n"
        f"MODEL: <code>{html_escape(LLM_MODEL)}</code>\n"
        f"OCR: <code>{'on' if OCR_PY_AVAILABLE else 'off'}</code>\n"
        f"FIGMA: <code>{figma_mode}</code>",
        reply_markup=keyboard,
    )


@dp.message(F.photo)
async def handle_photo(m: Message):
    chat_id = m.chat.id
    lock = get_chat_lock(chat_id)

    if lock.locked():
        await m.answer("Секунду. Я уже разбираю другой экран. Не смешиваем отчёты.", reply_markup=keyboard)
        return

    async with lock:
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

        await run_review_for_image(m, img, title="SCAN")


@dp.message(F.text.regexp(r"figma\.com/(file|design)/"))
async def handle_figma_link(m: Message):
    chat_id = m.chat.id
    lock = get_chat_lock(chat_id)

    if lock.locked():
        await m.answer("Секунду. Я уже разбираю другой экран. Не смешиваем отчёты.", reply_markup=keyboard)
        return

    async with lock:
        await run_review_for_figma_link(m, m.text or "")


@dp.message()
async def fallback(m: Message):
    await m.answer(
        "Я принимаю на ревью:\n"
        "• скриншоты (картинки)\n"
        "• ссылки на Figma фреймы (если файл публичный)\n\n"
        "Жми «Закинуть на ревью» или просто отправь скрин/ссылку.",
        reply_markup=keyboard,
    )


async def main():
    print(
        f"Design Review starting… model={LLM_MODEL}, OCR={OCR_PY_AVAILABLE}, CV={CV_AVAILABLE}, "
        f"Figma={'token' if FIGMA_TOKEN else 'public-only'}"
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
