# bot.py
# Design Review Buddy (screenshots + public Figma frame links)
# - Compact retro ASCII progress animation
# - 3 main menu buttons (Review / How it works / Channel)
# - Sends:
#   1) What I see
#   2) Verdict + recommendations (UX + text) + score /10
#   3) Annotated screenshot
#   4) Concept: always ASCII wireframe (retro). No "service unavailable" whining.
#
# Env vars:
#   BOT_TOKEN (required)
#   OPENAI_API_KEY (required)
#   LLM_MODEL (optional, default: gpt-4o-mini)
#   OCR_LANG (optional, default: rus+eng)
#   OCR_CONF_MIN (optional, default: 55)
#   MAX_VISION_SIDE (optional, default: 1280)

import asyncio
import base64
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import html as py_html
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    BufferedInputFile,
)

from PIL import Image, ImageDraw, ImageFont

import pytesseract
from openai import OpenAI


# =========================
# Config
# =========================
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

LLM_MODEL = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()

OCR_LANG = (os.getenv("OCR_LANG") or "rus+eng").strip()
OCR_CONF_MIN = int(os.getenv("OCR_CONF_MIN") or "55")

MAX_VISION_SIDE = int(os.getenv("MAX_VISION_SIDE") or "1280")

# progress animation
PROGRESS_DELAY = float(os.getenv("PROGRESS_DELAY") or "0.12")
PROGRESS_STEPS = int(os.getenv("PROGRESS_STEPS") or "22")

CHANNEL_URL = "https://t.me/prodooktovy"

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in environment variables (Railway Variables or local env).")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment variables (Railway Variables or local env).")

client = OpenAI(api_key=OPENAI_API_KEY)

router = Router()


# =========================
# UI copy
# =========================
BTN_REVIEW = "review"
BTN_HOW = "how"

WELCOME_TEXT = (
    "Я — партнёр для дизайн-ревью.\n\n"
    "Принимаю на разбор:\n"
    "• скриншоты интерфейса (картинки)\n"
    "• ссылки на Figma фреймы (если файл публичный)\n\n"
    "Жми «Закинуть на ревью» или просто отправь скрин/ссылку."
)

HOW_TEXT = (
    "Как это работает:\n"
    "1) Отправь скриншот или ссылку на публичный Figma-фрейм\n"
    "2) Я покажу прогресс обработки\n"
    "3) Верну:\n"
    "   • что я вижу\n"
    "   • вердикт и рекомендации (UX + текст) + оценка\n"
    "   • аннотации на скрине\n"
    "   • концепт (ASCII-вариант)"
)

REVIEW_HINT = (
    "Кидай сюда скриншот или ссылку на публичный Figma-фрейм.\n"
    "Я разберу и докопаюсь по делу 🙂 (без мата)."
)


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Закинуть на ревью", callback_data=BTN_REVIEW)],
            [InlineKeyboardButton(text="Как это работает?", callback_data=BTN_HOW)],
            [InlineKeyboardButton(text="Канал о продуктовом дизайне", url=CHANNEL_URL)],
        ]
    )


# =========================
# Helpers
# =========================
def html_escape(s: str) -> str:
    return py_html.escape(s, quote=False)

def is_probably_image_filename(name: str) -> bool:
    name = (name or "").lower()
    return any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"])

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def pil_open_image(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img

def resize_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)

def image_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def bytes_to_b64_data_url_png(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

async def safe_edit(msg: Message, text: str) -> Message:
    # Telegram sometimes forbids editing (e.g., message too old, race condition)
    try:
        await msg.edit_text(text, parse_mode=ParseMode.HTML)
        return msg
    except Exception:
        try:
            return await msg.answer(text, parse_mode=ParseMode.HTML)
        except Exception:
            return msg

def looks_like_figma_url(s: str) -> bool:
    s = (s or "").strip()
    return "figma.com" in s and ("node-id=" in s or "/design/" in s or "/file/" in s)

def normalize_figma_url(url: str) -> str:
    url = (url or "").strip()
    url = url.replace(" ", "")
    return url

def figma_oembed(url: str) -> Optional[Dict[str, Any]]:
    # No caching: always hit oEmbed for each link
    try:
        api = "https://www.figma.com/api/oembed?url=" + urllib.parse.quote(url, safe="")
        req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        return json.loads(data.decode("utf-8", errors="ignore"))
    except Exception:
        return None

def download_url_bytes(url: str, max_bytes: int = 8_000_000) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        return data
    except Exception:
        return None


# =========================
# Compact retro ASCII progress
# =========================
SPIN = ["|", "/", "-", "\\"]

def progress_line(step: int, total: int) -> str:
    # compact: 1 line, retro
    p = 0.0 if total <= 0 else step / float(total)
    bar_w = 16
    fill = int(p * bar_w)
    bar = "[" + ("#" * fill) + ("." * (bar_w - fill)) + "]"
    spin = SPIN[step % len(SPIN)]
    pct = int(p * 100)
    return f"{spin} {bar} {pct:>3d}%"

async def animate_progress(anchor: Message, title: str = "Review") -> Message:
    # One message, edited in-place; if edit fails -> fallback to send new.
    msg = await anchor.answer(f"<code>{html_escape(title)}\n{html_escape(progress_line(0, PROGRESS_STEPS))}</code>", parse_mode=ParseMode.HTML)
    for i in range(1, PROGRESS_STEPS + 1):
        await asyncio.sleep(PROGRESS_DELAY)
        msg = await safe_edit(msg, f"<code>{html_escape(title)}\n{html_escape(progress_line(i, PROGRESS_STEPS))}</code>")
    return msg


# =========================
# OCR extraction
# =========================
def extract_ocr_blocks(img: Image.Image) -> List[Dict[str, Any]]:
    # Use image_to_data for boxes
    data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    blocks: List[Dict[str, Any]] = []

    # Group words into lines by (block_num, par_num, line_num)
    groups: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf_raw = data.get("conf", ["0"])[i]
        try:
            conf = int(float(conf_raw))
        except Exception:
            conf = 0
        if not txt:
            continue
        if conf < OCR_CONF_MIN:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        groups.setdefault(key, []).append(i)

    for key, idxs in groups.items():
        # bbox union
        xs, ys, xe, ye = [], [], [], []
        words = []
        confs = []
        for i in idxs:
            x = int(data["left"][i]); y = int(data["top"][i])
            w = int(data["width"][i]); h = int(data["height"][i])
            xs.append(x); ys.append(y); xe.append(x + w); ye.append(y + h)
            words.append((data["text"][i] or "").strip())
            try:
                confs.append(int(float(data.get("conf", ["0"])[i])))
            except Exception:
                confs.append(0)
        text = " ".join(words).strip()
        if not text:
            continue
        x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
        blocks.append({
            "id": len(blocks),
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "conf": int(sum(confs) / max(1, len(confs))),
        })

    # Sort top-to-bottom
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    # reassign ids
    for i, b in enumerate(blocks):
        b["id"] = i
    return blocks


# =========================
# LLM: one call, strict JSON
# =========================
SCHEMA = {
    "name": "design_review_result",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "what_i_see": {"type": "string"},
            "score_10": {"type": "integer", "minimum": 0, "maximum": 10},
            "praise": {
                "type": "array",
                "items": {"type": "string"},
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "area": {"type": "string", "enum": ["ux", "text"]},
                        "severity": {"type": "integer", "minimum": 1, "maximum": 5},
                        "title": {"type": "string"},
                        "what_is_wrong": {"type": "string"},
                        "how_to_fix": {"type": "string"},
                        "block_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["area", "severity", "title", "what_is_wrong", "how_to_fix", "block_ids"],
                },
            },
            "ascii_concept": {"type": "string"},
        },
        "required": ["what_i_see", "score_10", "praise", "issues", "ascii_concept"],
    }
}

SYSTEM_PROMPT = (
    "Ты — жесткий, но адекватный старший дизайн-товарищ (без мата). "
    "Твоя цель — докопаться по делу и помочь улучшить UI и тексты. "
    "Не разводи воду. Если что-то ок — хвали конкретно. Если плохо — говори прямо. "
    "Про шрифты и палитру: только угадывай семейство/стиль (без размеров, без точных цветов). "
    "Контекст элементов (заголовок/кнопка/поле/подсказка) определяй аккуратно. "
    "Отдавай результат строго JSON по схеме."
)

def build_user_prompt(ocr_blocks: List[Dict[str, Any]], img_w: int, img_h: int) -> str:
    # Keep it short: send blocks list for referencing
    lines = []
    lines.append(f"Размер изображения: {img_w}x{img_h}.")
    lines.append("OCR-блоки (id, bbox[x1,y1,x2,y2], text):")
    for b in ocr_blocks[:120]:
        lines.append(f"{b['id']}: {b['bbox']} | {b['text']}")
    if len(ocr_blocks) > 120:
        lines.append(f"...и ещё {len(ocr_blocks)-120} блоков")
    lines.append(
        "Нужно:\n"
        "1) what_i_see: кратко, человечески, без JSON-оберток.\n"
        "2) score_10: честная оценка 0-10.\n"
        "3) praise: 0-5 пунктов.\n"
        "4) issues: список проблем (UX и Текст), с severity 1-5, и конкретным фиксом.\n"
        "   block_ids — привяжи к OCR-блокам, если возможно.\n"
        "5) ascii_concept: сделай ретро-вайрфрейм (ASCII), как могло бы быть лучше.\n"
    )
    return "\n".join(lines)

def call_llm_review(img_bytes: bytes, ocr_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    img = pil_open_image(img_bytes)
    w, h = img.size
    user_prompt = build_user_prompt(ocr_blocks, w, h)

    # Vision image (downscale for speed)
    vimg = resize_long_side(img, MAX_VISION_SIDE)
    vbuf = io.BytesIO()
    vimg.save(vbuf, format="PNG", optimize=True)
    vbytes = vbuf.getvalue()
    data_url = bytes_to_b64_data_url_png(vbytes)

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": SCHEMA["name"],
                "schema": SCHEMA["schema"],
                "strict": True,
            }
        },
    )

    # openai python: resp.output_text contains text; for json schema it will be json string
    raw = getattr(resp, "output_text", None)
    if not raw:
        # fallback: try to dig in outputs
        raw = ""
        try:
            for o in resp.output:
                for c in o.content:
                    if c.type in ("output_text", "text"):
                        raw += c.text
        except Exception:
            pass
    if not raw:
        raise RuntimeError("LLM returned empty response")

    return json.loads(raw)


# =========================
# Rendering output (HTML-safe)
# =========================
def format_verdict(result: Dict[str, Any]) -> str:
    score = result.get("score_10", 0)
    praise = result.get("praise", []) or []
    issues = result.get("issues", []) or []

    # Black/white emoji only: use simple bullets and symbols
    # No colored emojis.
    parts = []
    parts.append(f"<b>Оценка:</b> {int(score)}/10")
    if praise:
        parts.append("\n<b>Что хорошо:</b>")
        for p in praise[:6]:
            parts.append(f"• {html_escape(str(p))}")
    if issues:
        parts.append("\n<b>Что не ок и как исправить:</b>")
        # sort: severity desc, then area
        issues_sorted = sorted(issues, key=lambda x: (-int(x.get("severity", 1)), str(x.get("area", ""))))
        for it in issues_sorted[:14]:
            area = str(it.get("area", "ux")).upper()
            sev = int(it.get("severity", 1))
            title = html_escape(str(it.get("title", "")).strip())
            wrong = html_escape(str(it.get("what_is_wrong", "")).strip())
            fix = html_escape(str(it.get("how_to_fix", "")).strip())
            parts.append(f"\n<b>[{area}]</b> (жёсткость {sev}/5) — <b>{title}</b>\n{wrong}\n<b>Сделай так:</b> {fix}")
    else:
        parts.append("\n<b>Замечаний нет.</b> Подозрительно, но ладно 🙂")

    return "\n".join(parts).strip()

def format_what_i_see(result: Dict[str, Any]) -> str:
    s = str(result.get("what_i_see", "")).strip()
    return html_escape(s) if s else "Ничего внятного не вижу — попробуй прислать скрин крупнее."

def format_ascii_concept(result: Dict[str, Any]) -> str:
    concept = str(result.get("ascii_concept", "")).rstrip()
    if not concept:
        # always provide a minimal concept instead of whining
        concept = (
            "+----------------------+\n"
            "|  HEADER              |\n"
            "|  Subheader text      |\n"
            "|                      |\n"
            "|  [ Primary Action ]  |\n"
            "|  Secondary action    |\n"
            "+----------------------+\n"
        )
    # limit length to avoid telegram overflow
    concept = concept[:3000]
    return f"<code>{html_escape(concept)}</code>"


# =========================
# Annotations
# =========================
def draw_annotations(img_bytes: bytes, ocr_blocks: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> bytes:
    img = pil_open_image(img_bytes)
    draw = ImageDraw.Draw(img)

    # Use only b/w: black rectangles + white label background (simple)
    # Collect block ids to highlight with index numbers (1..N)
    seen: Dict[int, int] = {}
    label = 1
    for it in issues[:20]:
        for bid in (it.get("block_ids") or []):
            if isinstance(bid, int) and 0 <= bid < len(ocr_blocks) and bid not in seen:
                seen[bid] = label
                label += 1

    # Try a default font; if fails, use PIL default.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for bid, num in seen.items():
        x1, y1, x2, y2 = ocr_blocks[bid]["bbox"]
        # Expand a bit
        pad = 3
        x1 = clamp(x1 - pad, 0, img.size[0] - 1)
        y1 = clamp(y1 - pad, 0, img.size[1] - 1)
        x2 = clamp(x2 + pad, 0, img.size[0] - 1)
        y2 = clamp(y2 + pad, 0, img.size[1] - 1)

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=3)

        # label box
        tag = str(num)
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        bx1, by1 = x1, max(0, y1 - th - 6)
        bx2, by2 = x1 + tw + 10, y1
        draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0))
        draw.text((bx1 + 5, by1 + 2), tag, fill=(255, 255, 255), font=font)

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


# =========================
# Processing: image / figma link
# =========================
async def process_and_reply(anchor: Message, img_bytes: bytes, source_title: str = "Screenshot") -> None:
    # Progress animation
    await animate_progress(anchor, title="REVIEW")

    # OCR
    img = pil_open_image(img_bytes)
    ocr_blocks = extract_ocr_blocks(img)

    # LLM Review
    result = call_llm_review(img_bytes, ocr_blocks)

    # Message 1: what I see
    await anchor.answer(
        f"<b>Что вижу:</b>\n{format_what_i_see(result)}",
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu(),
    )

    # Message 2: verdict (ux+text together)
    await anchor.answer(
        format_verdict(result),
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu(),
    )

    # Message 3: annotated screenshot (if we have blocks/issues)
    issues = result.get("issues", []) or []
    annotated_bytes = draw_annotations(img_bytes, ocr_blocks, issues)
    await anchor.answer_photo(
        BufferedInputFile(annotated_bytes, filename="annotated.png"),
        caption="Аннотации (цифры = места, где есть что поправить)",
        reply_markup=main_menu(),
    )

    # Small progress between 3 and 4 (compact)
    await animate_progress(anchor, title="CONCEPT")

    # Message 4: concept (always ASCII; no failure text)
    await anchor.answer(
        f"<b>Концепт (ASCII):</b>\n{format_ascii_concept(result)}",
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu(),
    )


async def process_figma_link(anchor: Message, url: str) -> None:
    url = normalize_figma_url(url)
    o = figma_oembed(url)
    if not o:
        await anchor.answer(
            "Не смог получить превью по ссылке. Проверь, что файл публичный, и пришли ссылку ещё раз.",
            reply_markup=main_menu(),
        )
        return

    thumb = o.get("thumbnail_url") or o.get("thumbnail") or ""
    title = (o.get("title") or "Figma frame").strip()

    if not thumb:
        await anchor.answer(
            "Я вижу ссылку, но превью фигмы не отдало картинку. Проверь публичность файла.",
            reply_markup=main_menu(),
        )
        return

    img_bytes = download_url_bytes(thumb, max_bytes=MAX_PREVIEW_BYTES)
    if not img_bytes:
        await anchor.answer(
            "Не смог скачать превью (слишком большое или недоступно). Попробуй другую ссылку/крупнее.",
            reply_markup=main_menu(),
        )
        return

    # Show preview first
    try:
        await anchor.answer_photo(
            BufferedInputFile(img_bytes, filename="figma_preview.png"),
            caption=f"Превью из Figma: {title}",
            reply_markup=main_menu(),
        )
    except Exception:
        # ok, continue anyway
        pass

    await process_and_reply(anchor, img_bytes, source_title=title)


# =========================
# Handlers: start/menu
# =========================
@router.message(F.text == "/start")
async def on_start(m: Message) -> None:
    await m.answer(WELCOME_TEXT, reply_markup=main_menu())

@router.callback_query(F.data == BTN_REVIEW)
async def on_review_btn(c: CallbackQuery) -> None:
    await c.answer()
    await c.message.answer(REVIEW_HINT, reply_markup=main_menu())

@router.callback_query(F.data == BTN_HOW)
async def on_how_btn(c: CallbackQuery) -> None:
    await c.answer()
    await c.message.answer(HOW_TEXT, reply_markup=main_menu())


# =========================
# Handlers: images
# =========================
@router.message(F.photo)
async def on_photo(m: Message) -> None:
    try:
        ph = m.photo[-1]
        file = await m.bot.get_file(ph.file_id)
        buf = io.BytesIO()
        await m.bot.download_file(file.file_path, buf)
        img_bytes = buf.getvalue()
        await process_and_reply(m, img_bytes, source_title="Screenshot")
    except Exception as e:
        await m.answer(f"Сломался на картинке: {html_escape(str(e))}", reply_markup=main_menu(), parse_mode=ParseMode.HTML)

@router.message(F.document)
async def on_document(m: Message) -> None:
    doc = m.document
    if not doc:
        return
    if not is_probably_image_filename(doc.file_name or ""):
        await m.answer("Это не похоже на картинку. Пришли PNG/JPG или ссылку на Figma.", reply_markup=main_menu())
        return
    try:
        file = await m.bot.get_file(doc.file_id)
        buf = io.BytesIO()
        await m.bot.download_file(file.file_path, buf)
        img_bytes = buf.getvalue()
        await process_and_reply(m, img_bytes, source_title="Screenshot")
    except Exception as e:
        await m.answer(f"Сломался на файле: {html_escape(str(e))}", reply_markup=main_menu(), parse_mode=ParseMode.HTML)


# =========================
# Handlers: text links
# =========================
FIGMA_URL_RE = re.compile(r"(https?://[^\s]+figma\.com[^\s]+)", re.IGNORECASE)

@router.message(F.text)
async def on_text(m: Message) -> None:
    txt = (m.text or "").strip()
    if not txt:
        await m.answer(WELCOME_TEXT, reply_markup=main_menu())
        return

    # Find figma link in message
    match = FIGMA_URL_RE.search(txt)
    if match and looks_like_figma_url(match.group(1)):
        await process_figma_link(m, match.group(1))
        return

    # fallback
    await m.answer(WELCOME_TEXT, reply_markup=main_menu())


# =========================
# Main
# =========================
async def main() -> None:
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    print(f"✅ Bot starting... OCR_LANG={OCR_LANG}, model={LLM_MODEL}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())