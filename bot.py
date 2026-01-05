#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Design Reviewer Telegram Bot (aiogram v3)

What it does:
- Accepts screenshots (photo/document image) OR public Figma frame links
- Shows retro ASCII progress animation while processing
- Allows cancel during processing
- Sends 4 outputs per review:
  1) What I see on the screen
  2) Verdict + recommendations (UX + Text) + score /10
  3) Annotated screenshot (numbered callouts)
  4) ASCII concept alternative (retro wireframe)

Important UX rules implemented:
- Main menu buttons appear ONLY:
    - after /start
    - at the END of each review (or after error/cancel)
- During processing: only "Cancel" inline button is shown
- No dotenv, no requests, no httpx (Railway-friendly)
- Avoids Telegram HTML parsing problems by sending plain text (no parse_mode)
"""

import asyncio
import base64
import io
import json
import os
import re
import time
import html as py_html
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import pytesseract

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ContentType
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
    BotCommand,
)
from aiogram.client.default import DefaultBotProperties

# Optional LLM
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# ENV / CONFIG
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN environment variable.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_ENABLED = (os.getenv("LLM_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
OCR_LANG = os.getenv("OCR_LANG", "rus+eng").strip()

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PREVIEW_BYTES = 8 * 1024 * 1024

# Timeouts (seconds)
OCR_TIMEOUT = 25
LLM_TIMEOUT = 65
FIGMA_OEMBED_TIMEOUT = 15
FIGMA_DOWNLOAD_TIMEOUT = 20

# Callback
BTN_CANCEL = "cancel_review"

# Router
router = Router()

# Active review cancel events per chat
active_reviews: Dict[int, asyncio.Event] = {}


# =========================
# UI: MAIN MENU
# =========================
def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="1) Send for review (screenshot/link)")],
            [KeyboardButton(text="2) How it works?")],
            [KeyboardButton(text="3) Design channel @prodooktovy")],
        ],
        resize_keyboard=True,
        is_persistent=True,
        selective=False,
    )


def cancel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Cancel", callback_data=BTN_CANCEL)]
        ]
    )


# =========================
# ASCII ANIMATION (KEEP IT FUN, NOT HUGE)
# =========================
SPIN = ["|", "/", "-", "\\"]

def ascii_frame(step: int, title: str) -> str:
    # compact retro block
    bar_w = 22
    fill = step % (bar_w + 1)
    bar = "[" + ("#" * fill) + ("." * (bar_w - fill)) + "]"
    return f"{SPIN[step % 4]} {title}\n{bar}"


async def safe_edit(msg: Message, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> Message:
    try:
        return await msg.edit_text(text, reply_markup=reply_markup)
    except Exception:
        return msg


async def animate_progress(anchor: Message, title: str, done_evt: asyncio.Event, cancel_markup: InlineKeyboardMarkup) -> Message:
    # Send first message with cancel button
    m = await anchor.answer(f"```text\n{ascii_frame(0, title)}\n```", reply_markup=cancel_markup)
    step = 1
    while not done_evt.is_set():
        await asyncio.sleep(0.18)
        # keep updating while allowed
        await safe_edit(m, f"```text\n{ascii_frame(step, title)}\n```", reply_markup=cancel_markup)
        step += 1
    return m


# =========================
# UTIL
# =========================
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def is_probably_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def looks_like_figma_url(s: str) -> bool:
    return "figma.com/" in s and ("node-id=" in s or "/design/" in s or "/file/" in s)


def normalize_figma_url(url: str) -> str:
    # Keep node-id & other params intact; just strip whitespace
    return url.strip()


def pil_open_image(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    # Convert to RGB for drawing + OCR
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img


def download_url_bytes(url: str, max_bytes: int) -> Optional[bytes]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (DesignReviewerBot/1.0)"
        }
    )
    with urllib.request.urlopen(req, timeout=FIGMA_DOWNLOAD_TIMEOUT) as r:
        data = r.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        return data


def figma_oembed(figma_url: str) -> Optional[Dict[str, Any]]:
    oembed = "https://www.figma.com/api/oembed?url=" + urllib.parse.quote(figma_url, safe="")
    req = urllib.request.Request(
        oembed,
        headers={"User-Agent": "Mozilla/5.0 (DesignReviewerBot/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=FIGMA_OEMBED_TIMEOUT) as r:
        raw = r.read(512 * 1024)
        return json.loads(raw.decode("utf-8"))


# =========================
# OCR
# =========================
def extract_ocr_blocks(img: Image.Image) -> List[Dict[str, Any]]:
    # Use image_to_data to get boxes
    data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    blocks: List[Dict[str, Any]] = []

    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if not txt:
            continue
        # Heuristic: ignore garbage / ultra-low confidence
        if conf != -1.0 and conf < 35:
            continue

        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        blocks.append({
            "id": i,
            "text": txt,
            "conf": conf,
            "bbox": [x, y, w, h],
            "line": int(data.get("line_num", [0]*n)[i]) if "line_num" in data else 0,
            "block": int(data.get("block_num", [0]*n)[i]) if "block_num" in data else 0,
            "par": int(data.get("par_num", [0]*n)[i]) if "par_num" in data else 0,
        })

    # Merge words into line chunks
    merged = merge_blocks_to_lines(blocks)
    return merged


def merge_blocks_to_lines(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []

    # Group by (block, par, line)
    groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for w in words:
        key = (w.get("block", 0), w.get("par", 0), w.get("line", 0))
        groups.setdefault(key, []).append(w)

    lines: List[Dict[str, Any]] = []
    for _, items in groups.items():
        items = sorted(items, key=lambda z: z["bbox"][0])
        text = " ".join([it["text"] for it in items]).strip()
        if not text:
            continue
        xs = [it["bbox"][0] for it in items]
        ys = [it["bbox"][1] for it in items]
        x2 = [it["bbox"][0] + it["bbox"][2] for it in items]
        y2 = [it["bbox"][1] + it["bbox"][3] for it in items]
        bbox = [min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys)]
        confs = [it.get("conf", -1.0) for it in items if isinstance(it.get("conf", None), (int, float))]
        conf_avg = sum(confs) / max(1, len(confs))
        word_ids = [it["id"] for it in items]
        lines.append({
            "text": text,
            "bbox": bbox,
            "conf": conf_avg,
            "word_ids": word_ids,
        })

    # Sort top-to-bottom
    lines.sort(key=lambda z: (z["bbox"][1], z["bbox"][0]))
    return lines


# =========================
# ANNOTATIONS
# =========================
def draw_annotations(img_bytes: bytes, lines: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> bytes:
    img = pil_open_image(img_bytes)
    draw = ImageDraw.Draw(img)

    # Font (optional)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # We annotate by "target_index" (line index in OCR lines)
    for k, iss in enumerate(issues, start=1):
        idx = iss.get("target_index", None)
        if idx is None:
            continue
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(lines):
            continue

        x, y, w, h = lines[idx]["bbox"]
        # Draw rectangle (thick-ish)
        for t in range(3):
            draw.rectangle([x - t, y - t, x + w + t, y + h + t], outline=(0, 0, 0))
        # Number label (small filled box)
        label = str(k)
        lx = clamp(x - 6, 0, img.size[0]-1)
        ly = clamp(y - 28, 0, img.size[1]-1)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 6
        draw.rectangle([lx, ly, lx + tw + pad*2, ly + th + pad*2], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((lx + pad, ly + pad), label, fill=(0, 0, 0), font=font)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# =========================
# LLM (ONE CALL) + FALLBACK
# =========================
def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    # try find first { ... } block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def call_llm_review(img_bytes: bytes, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    # If no OpenAI or disabled -> heuristic fallback
    if not LLM_ENABLED or not OPENAI_API_KEY or OpenAI is None:
        return heuristic_review(lines)

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare OCR summary for the model (limited)
    ocr_list = []
    for i, ln in enumerate(lines[:80]):
        ocr_list.append({"i": i, "text": ln["text"]})

    prompt = f"""
You are a strict-but-fair senior product designer reviewing a UI screenshot.

Rules:
- Be blunt, no insults, no profanity.
- If something is good, praise it with specifics.
- For fonts/palette: only GUESS font family vibes (e.g., "Inter-like", "SF Pro-like"). No sizes, no numeric colors.
- Output must be understandable and actionable: always say what is wrong AND what to do instead.
- Score 0..10 (integer).
- Give issues that map to OCR line indices when possible.

Return JSON (no markdown) in this exact shape:
{{
  "what_i_see": "plain text",
  "score": 7,
  "verdict": {{
    "ux": ["bullet 1", "bullet 2"],
    "text": ["bullet 1", "bullet 2"],
    "praise": ["bullet 1", "bullet 2"]
  }},
  "issues": [
    {{
      "target_index": 3,
      "title": "Short label",
      "problem": "What's wrong",
      "fix": "What to do",
      "severity": "low|medium|high"
    }}
  ],
  "concept_ascii": "ASCII wireframe concept (max 18 lines)"
}}

OCR lines (index -> text):
{json.dumps(ocr_list, ensure_ascii=False)}
""".strip()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = "data:image/png;base64," + img_b64

    # Responses API content types MUST be input_text / input_image
    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    text = getattr(resp, "output_text", None) or ""
    parsed = extract_first_json(text)
    if not parsed:
        # fallback: try use raw text
        return heuristic_review(lines, note="LLM returned non-JSON; used fallback.")

    # sanitize
    return normalize_llm_output(parsed, lines)


def normalize_llm_output(d: Dict[str, Any], lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["what_i_see"] = str(d.get("what_i_see", "")).strip()
    # score
    try:
        sc = int(d.get("score", 0))
    except Exception:
        sc = 0
    out["score"] = clamp(sc, 0, 10)

    verdict = d.get("verdict", {}) if isinstance(d.get("verdict", {}), dict) else {}
    out["verdict"] = {
        "ux": [str(x).strip() for x in (verdict.get("ux", []) or [])][:12],
        "text": [str(x).strip() for x in (verdict.get("text", []) or [])][:12],
        "praise": [str(x).strip() for x in (verdict.get("praise", []) or [])][:12],
    }

    issues_in = d.get("issues", []) if isinstance(d.get("issues", []), list) else []
    issues: List[Dict[str, Any]] = []
    for it in issues_in[:20]:
        if not isinstance(it, dict):
            continue
        idx = it.get("target_index", None)
        if isinstance(idx, int) and 0 <= idx < len(lines):
            pass
        else:
            # allow None
            idx = None
        sev = str(it.get("severity", "medium")).lower().strip()
        if sev not in ("low", "medium", "high"):
            sev = "medium"
        issues.append({
            "target_index": idx,
            "title": str(it.get("title", "")).strip(),
            "problem": str(it.get("problem", "")).strip(),
            "fix": str(it.get("fix", "")).strip(),
            "severity": sev,
        })
    out["issues"] = issues

    concept = str(d.get("concept_ascii", "")).rstrip()
    if not concept:
        concept = default_ascii_concept()
    out["concept_ascii"] = trim_ascii(concept, max_lines=18, max_width=48)

    return out


def trim_ascii(s: str, max_lines: int, max_width: int) -> str:
    lines = s.splitlines()
    lines = lines[:max_lines]
    lines = [ln[:max_width] for ln in lines]
    return "\n".join(lines).strip()


def default_ascii_concept() -> str:
    return "\n".join([
        "+--------------------------------------+",
        "|  Title                                |",
        "|  Short explanation                    |",
        "|                                      |",
        "|  [ Primary action ]                   |",
        "|  [ Secondary ]                        |",
        "|                                      |",
        "|  Hint / helper text                   |",
        "+--------------------------------------+",
    ])


def heuristic_review(lines: List[Dict[str, Any]], note: str = "") -> Dict[str, Any]:
    # Basic, but not useless: give something actionable.
    all_text = " ".join([l["text"] for l in lines]).lower()
    praise = []
    ux = []
    tx = []
    issues = []

    if len(lines) >= 3:
        praise.append("Есть структурированные текстовые блоки — экран, вероятно, не «каша».")

    if any("успеш" in all_text for _ in [0]):
        tx.append("Слово «успешно» часто раздувает статус. Лучше писать факт: «Отправили», «Создали», «Готово».")

    if any("ошиб" in all_text for _ in [0]):
        tx.append("Если есть ошибка — добавь конкретику: что случилось и что делать дальше (следующий шаг).")

    ux.append("Проверь визуальную иерархию: один главный акцент, остальное — поддержка.")
    ux.append("Если на экране много текста — сделай короткий заголовок + 1–2 строки, остальное в детали.")

    # Make a couple issues mapped to first lines (if any)
    if lines:
        issues.append({
            "target_index": 0,
            "title": "Слишком расплывчато",
            "problem": "Текст звучит общо — пользователь может не понять, что произойдёт дальше.",
            "fix": "Перепиши как факт + следующий шаг: «Мы делаем X. Результат будет Y. Проверить в разделе Z».",
            "severity": "medium",
        })

    if note:
        tx.append(note)

    return {
        "what_i_see": "Вижу экран интерфейса с текстовыми элементами. Без LLM могу оценить только общие риски.",
        "score": 6,
        "verdict": {"ux": ux, "text": tx, "praise": praise},
        "issues": issues,
        "concept_ascii": default_ascii_concept(),
    }


# =========================
# FORMATTING (NO HTML, NO TAGS)
# =========================
def fmt_bullets(items: List[str], prefix: str = "- ") -> str:
    return "\n".join([prefix + x for x in items if x.strip()])


def format_what_i_see(r: Dict[str, Any]) -> str:
    s = (r.get("what_i_see") or "").strip()
    if not s:
        s = "I see a UI screen with text blocks and interface controls."
    return s


def format_verdict(r: Dict[str, Any]) -> str:
    score = r.get("score", 0)
    v = r.get("verdict", {}) if isinstance(r.get("verdict", {}), dict) else {}
    ux = v.get("ux", []) if isinstance(v.get("ux", []), list) else []
    tx = v.get("text", []) if isinstance(v.get("text", []), list) else []
    pr = v.get("praise", []) if isinstance(v.get("praise", []), list) else []

    parts = []
    parts.append(f"VERDICT: {score}/10")
    if pr:
        parts.append("\nGOOD (keep it):\n" + fmt_bullets(pr))
    if ux:
        parts.append("\nUX (fix it):\n" + fmt_bullets(ux))
    if tx:
        parts.append("\nTEXT (fix it):\n" + fmt_bullets(tx))

    # Issues list (numbered)
    issues = r.get("issues", []) if isinstance(r.get("issues", []), list) else []
    if issues:
        lines = []
        for i, it in enumerate(issues[:12], start=1):
            title = (it.get("title") or "Issue").strip()
            sev = (it.get("severity") or "medium").strip().lower()
            problem = (it.get("problem") or "").strip()
            fix = (it.get("fix") or "").strip()
            lines.append(f"{i}) [{sev}] {title}\n   - Problem: {problem}\n   - Fix: {fix}")
        parts.append("\nCALLOUTS:\n" + "\n".join(lines))

    return "\n".join(parts).strip()


def format_ascii_concept(r: Dict[str, Any]) -> str:
    s = (r.get("concept_ascii") or "").rstrip()
    if not s:
        s = default_ascii_concept()
    return s


# =========================
# CORE PROCESSING (CANCEL + NO MENU SPAM)
# =========================
async def ocr_lines_async(img: Image.Image) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(extract_ocr_blocks, img)


async def llm_review_async(img_bytes: bytes, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    return await asyncio.to_thread(call_llm_review, img_bytes, lines)


async def process_image_review(anchor: Message, img_bytes: bytes, source_label: str = "Screenshot") -> None:
    chat_id = anchor.chat.id
    cancel_evt = asyncio.Event()
    active_reviews[chat_id] = cancel_evt

    done_evt = asyncio.Event()
    spinner_task = asyncio.create_task(
        animate_progress(anchor, f"REVIEW {source_label}", done_evt, cancel_keyboard())
    )

    try:
        img = pil_open_image(img_bytes)

        # OCR
        if cancel_evt.is_set():
            raise asyncio.CancelledError()

        lines = await asyncio.wait_for(ocr_lines_async(img), timeout=OCR_TIMEOUT)

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

        # LLM review (optional)
        review = await asyncio.wait_for(llm_review_async(img_bytes, lines), timeout=LLM_TIMEOUT)

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

    except asyncio.CancelledError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        # end with menu
        await anchor.answer("Cancelled. Send another screenshot or a public Figma frame link.", reply_markup=main_menu())
        return

    except asyncio.TimeoutError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(
            "Timed out. Send the same screen again (preferably larger / clearer).",
            reply_markup=main_menu(),
        )
        return

    except Exception as e:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(f"Processing failed: {str(e)}", reply_markup=main_menu())
        return

    finally:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        # clear active review
        if chat_id in active_reviews:
            del active_reviews[chat_id]

    # 1) what i see (NO menu)
    await anchor.answer("WHAT I SEE:\n" + format_what_i_see(review))

    # 2) verdict + recommendations (NO menu)
    await anchor.answer(format_verdict(review))

    # 3) annotated screenshot (NO menu)
    try:
        annotated = await asyncio.to_thread(draw_annotations, img_bytes, lines, review.get("issues", []))
        await anchor.answer_photo(
            BufferedInputFile(annotated, filename="annotated.png"),
            caption="ANNOTATIONS: numbers match CALLOUTS list above.",
        )
    except Exception:
        await anchor.answer("Annotations failed this time. The written feedback still applies.")

    # extra retro spinner before concept (small)
    done2 = asyncio.Event()
    spinner2 = asyncio.create_task(animate_progress(anchor, "CONCEPT", done2, cancel_keyboard()))
    await asyncio.sleep(0.9)
    done2.set()
    try:
        await spinner2
    except Exception:
        pass

    # 4) concept (NO menu)
    await anchor.answer("CONCEPT (ASCII):\n```text\n" + format_ascii_concept(review) + "\n```")

    # end with menu (ONLY HERE)
    await anchor.answer("Done. Send another screenshot or a public Figma frame link.", reply_markup=main_menu())


async def process_figma_link(anchor: Message, url: str) -> None:
    chat_id = anchor.chat.id
    cancel_evt = asyncio.Event()
    active_reviews[chat_id] = cancel_evt

    done_evt = asyncio.Event()
    spinner_task = asyncio.create_task(
        animate_progress(anchor, "FETCH FIGMA PREVIEW", done_evt, cancel_keyboard())
    )

    try:
        figma_url = normalize_figma_url(url)

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

        o = await asyncio.wait_for(asyncio.to_thread(figma_oembed, figma_url), timeout=FIGMA_OEMBED_TIMEOUT)
        if not o:
            raise RuntimeError("Figma oEmbed returned empty.")

        thumb = o.get("thumbnail_url") or o.get("thumbnail") or ""
        title = (o.get("title") or "Figma frame").strip()
        if not thumb:
            raise RuntimeError("No thumbnail_url from Figma. Make sure the file is public.")

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

        img_bytes = await asyncio.wait_for(asyncio.to_thread(download_url_bytes, thumb, MAX_PREVIEW_BYTES), timeout=FIGMA_DOWNLOAD_TIMEOUT)
        if not img_bytes:
            raise RuntimeError("Failed to download preview (too large / blocked).")

    except asyncio.CancelledError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer("Cancelled.", reply_markup=main_menu())
        return
    except asyncio.TimeoutError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer("Figma fetch timed out. Try again.", reply_markup=main_menu())
        return
    except Exception as e:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(f"Could not fetch Figma preview: {str(e)}", reply_markup=main_menu())
        return
    finally:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        if chat_id in active_reviews:
            del active_reviews[chat_id]

    # Show the preview image (proof we fetched the correct node)
    await anchor.answer_photo(
        BufferedInputFile(img_bytes, filename="figma_preview.png"),
        caption="Figma preview fetched. Reviewing now...",
    )

    # Now run normal image review pipeline
    await process_image_review(anchor, img_bytes, source_label="Figma")


# =========================
# HANDLERS
# =========================
@router.message(Command("start"))
async def on_start(m: Message):
    text = (
        "Design Reviewer.\n\n"
        "I accept for review:\n"
        "- screenshots (images)\n"
        "- public Figma frame links (if the file is public)\n\n"
        "Use the menu or just send a screenshot/link."
    )
    await m.answer(text, reply_markup=main_menu())


@router.callback_query(F.data == BTN_CANCEL)
async def on_cancel(cb: CallbackQuery):
    chat_id = cb.message.chat.id if cb.message else None
    if chat_id is not None:
        evt = active_reviews.get(chat_id)
        if evt:
            evt.set()
    # Try to edit the spinner message (best effort)
    try:
        if cb.message:
            await cb.message.edit_text("Cancelling…")
    except Exception:
        pass
    await cb.answer("Cancelled")


@router.message(F.text == "2) How it works?")
async def on_help(m: Message):
    await m.answer(
        "How it works:\n"
        "1) Send a screenshot OR a public Figma frame link\n"
        "2) I analyze what’s on screen\n"
        "3) You get: what I see + verdict + annotations + ASCII concept\n\n"
        "Tip: clearer screenshots = better feedback.",
        # no menu spam; keep menu already visible from /start
    )


@router.message(F.text == "3) Design channel @prodooktovy")
async def on_channel(m: Message):
    # Telegram doesn't allow "subscribe" button universally; simplest is link button
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Open @prodooktovy", url="https://t.me/prodooktovy")]
        ]
    )
    await m.answer("Design channel:", reply_markup=kb)


@router.message(F.text == "1) Send for review (screenshot/link)")
async def on_send_for_review_hint(m: Message):
    await m.answer("Send a screenshot image or paste a public Figma frame link.")


@router.message(F.content_type == ContentType.PHOTO)
async def on_photo(m: Message, bot: Bot):
    # Take the biggest photo
    ph = m.photo[-1]
    file = await bot.get_file(ph.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    img_bytes = buf.getvalue()
    if len(img_bytes) > MAX_IMAGE_BYTES:
        await m.answer("Image too large. Send a smaller screenshot.", reply_markup=main_menu())
        return
    await process_image_review(m, img_bytes, source_label="Screenshot")


@router.message(F.content_type == ContentType.DOCUMENT)
async def on_document(m: Message, bot: Bot):
    # Accept only images as documents
    doc = m.document
    if not doc:
        return
    mime = (doc.mime_type or "").lower()
    if not (mime.startswith("image/") or doc.file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        await m.answer("Send an image file (PNG/JPG) or a Figma link.", reply_markup=main_menu())
        return

    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    img_bytes = buf.getvalue()
    if len(img_bytes) > MAX_IMAGE_BYTES:
        await m.answer("Image too large. Send a smaller screenshot.", reply_markup=main_menu())
        return
    await process_image_review(m, img_bytes, source_label="Screenshot")


@router.message(F.text)
async def on_text(m: Message):
    t = (m.text or "").strip()
    if looks_like_figma_url(t) and looks_like_figma_url(t):
        await process_figma_link(m, t)
        return

    # If user pasted other links or text
    if is_probably_url(t):
        await m.answer(
            "I can review:\n- screenshots (images)\n- public Figma frame links (file must be public)\n\n"
            "Send a screenshot or a Figma link with node-id.",
            # keep menu only if we need to recover the UI
            reply_markup=main_menu(),
        )
        return

    # Random text
    await m.answer(
        "Send a screenshot or a public Figma frame link. Text alone isn’t enough for this bot.",
        # menu already usually visible
    )


# =========================
# MAIN
# =========================
async def main():
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=None)  # avoid HTML entity failures
    )
    dp = Dispatcher()
    dp.include_router(router)

    # Set bot commands (nice to have)
    try:
        await bot.set_my_commands([BotCommand(command="start", description="Start")])
    except Exception:
        pass

    # Run
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())