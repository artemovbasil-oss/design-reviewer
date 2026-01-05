#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Design Reviewer Telegram Bot (aiogram v3)

- Accepts screenshots OR public Figma frame links
- Retro ASCII progress animation (monospace)
- Cancel button during processing
- 4 outputs per review:
  1) What I see
  2) Verdict + recommendations (UX + Text) + score /10
  3) Annotated screenshot (numbered callouts)
  4) ASCII concept alternative (monospace)

UX rules:
- Main menu buttons appear ONLY:
    - after /start
    - at the END of each review (or after error/cancel)
- During processing: only "Cancel" inline button

Languages:
- Default: EN
- Toggle via menu button "Language: EN/RU"
"""

import asyncio
import base64
import io
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import pytesseract

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ContentType, ParseMode
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

# Optional OpenAI SDK
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
LLM_TIMEOUT = 75
FIGMA_OEMBED_TIMEOUT = 15
FIGMA_DOWNLOAD_TIMEOUT = 20

# Callback
BTN_CANCEL = "cancel_review"

# Router
router = Router()

# Active review cancel events per chat
active_reviews: Dict[int, asyncio.Event] = {}

# Per-chat language (default EN)
chat_lang: Dict[int, str] = {}  # "EN" / "RU"


# =========================
# I18N
# =========================
def lang_of(chat_id: int) -> str:
    return chat_lang.get(chat_id, "EN")


def t(chat_id: int, key: str) -> str:
    L = lang_of(chat_id)
    EN = {
        "start_title": "Design Reviewer.",
        "start_body": (
            "I accept for review:\n"
            "- screenshots (images)\n"
            "- public Figma frame links (if the file is public)\n\n"
            "Use the menu or just send a screenshot/link."
        ),
        "btn_send": "1) Send for review (screenshot/link)",
        "btn_how": "2) How it works?",
        "btn_channel": "3) Product Design channel @prodooktovy",
        "btn_lang_en": "Language: EN (tap to switch)",
        "btn_lang_ru": "Язык: RU (нажми, чтобы переключить)",
        "how": (
            "How it works:\n"
            "1) Send a screenshot OR a public Figma frame link\n"
            "2) I analyze what’s on screen\n"
            "3) You get: what I see + verdict + annotations + ASCII concept\n\n"
            "Tip: clearer screenshots = better feedback."
        ),
        "send_hint": "Send a screenshot image or paste a public Figma frame link.",
        "channel": "Product Design channel:",
        "cancelled": "Cancelled. Send another screenshot or a public Figma frame link.",
        "timeout": "Timed out. Send the same screen again (preferably larger / clearer).",
        "img_too_large": "Image too large. Send a smaller screenshot.",
        "doc_need_img": "Send an image file (PNG/JPG) or a Figma link.",
        "figma_bad_link": (
            "I can review:\n- screenshots (images)\n- public Figma frame links (file must be public)\n\n"
            "Send a screenshot or a Figma link with node-id."
        ),
        "text_alone": "Send a screenshot or a public Figma frame link. Text alone isn’t enough.",
        "done": "Done. Send another screenshot or a public Figma frame link.",
        "what_i_see": "WHAT I SEE:",
        "annotations_caption": "ANNOTATIONS: numbers match CALLOUTS list above.",
        "concept_title": "CONCEPT (ASCII):",
        "figma_preview_caption": "Figma preview fetched. Reviewing now...",
        "fetch_failed": "Could not fetch Figma preview: ",
        "fetch_timeout": "Figma fetch timed out. Try again.",
        "processing_failed": "Processing failed: ",
        "llm_disabled": "LLM is disabled; using fallback review.",
        "llm_missing_key": "OPENAI_API_KEY is missing; using fallback review.",
        "llm_sdk_missing": "OpenAI SDK not available; using fallback review.",
        "llm_error_prefix": "LLM error; using fallback review: ",
        "verdict_title": "VERDICT:",
        "good": "GOOD (keep it):",
        "ux": "UX (fix it):",
        "text": "TEXT (fix it):",
        "callouts": "CALLOUTS:",
        "annot_fail": "Annotations failed this time. The written feedback still applies.",
    }
    RU = {
        "start_title": "Design Reviewer.",
        "start_body": (
            "Я принимаю на ревью:\n"
            "- скриншоты (картинки)\n"
            "- публичные ссылки на Figma фреймы (если файл публичный)\n\n"
            "Жми кнопки или просто отправь скрин/ссылку."
        ),
        "btn_send": "1) Закинуть на ревью (скрин/ссылка)",
        "btn_how": "2) Как это работает?",
        "btn_channel": "3) Канал о проддизайне @prodooktovy",
        "btn_lang_en": "Language: EN (tap to switch)",
        "btn_lang_ru": "Язык: RU (нажми, чтобы переключить)",
        "how": (
            "Как это работает:\n"
            "1) Отправь скриншот ИЛИ публичную ссылку на Figma фрейм\n"
            "2) Я разберу, что на экране\n"
            "3) Ты получишь: что вижу + вердикт + аннотации + ASCII-концепт\n\n"
            "Совет: чем четче скрин — тем полезнее ревью."
        ),
        "send_hint": "Отправь картинку-скриншот или вставь публичную ссылку на фрейм Figma.",
        "channel": "Канал о продуктовом дизайне:",
        "cancelled": "Отменено. Отправь новый скриншот или публичную Figma-ссылку.",
        "timeout": "Таймаут. Пришли тот же скрин ещё раз (лучше крупнее/четче).",
        "img_too_large": "Картинка слишком большая. Пришли меньший файл.",
        "doc_need_img": "Нужен файл-картинка (PNG/JPG) или ссылка на Figma.",
        "figma_bad_link": (
            "Я могу ревьюить:\n- скриншоты (картинки)\n- ссылки на Figma фреймы (если файл публичный)\n\n"
            "Пришли скриншот или Figma ссылку с node-id."
        ),
        "text_alone": "Нужен скриншот или публичная ссылка на Figma. Текст сам по себе — мало.",
        "done": "Готово. Отправь следующий скриншот или публичную Figma-ссылку.",
        "what_i_see": "ЧТО Я ВИЖУ:",
        "annotations_caption": "АННОТАЦИИ: номера соответствуют списку CALLOUTS выше.",
        "concept_title": "КОНЦЕПТ (ASCII):",
        "figma_preview_caption": "Превью из Figma получено. Делаю ревью...",
        "fetch_failed": "Не смог скачать превью из Figma: ",
        "fetch_timeout": "Таймаут при получении Figma. Попробуй снова.",
        "processing_failed": "Ошибка обработки: ",
        "llm_disabled": "LLM выключен; использую fallback-ревью.",
        "llm_missing_key": "Нет OPENAI_API_KEY; использую fallback-ревью.",
        "llm_sdk_missing": "Нет OpenAI SDK; использую fallback-ревью.",
        "llm_error_prefix": "Ошибка LLM; fallback-ревью: ",
        "verdict_title": "ВЕРДИКТ:",
        "good": "ХОРОШО (сохрани):",
        "ux": "UX (починить):",
        "text": "ТЕКСТ (починить):",
        "callouts": "CALLOUTS:",
        "annot_fail": "Аннотации не получились. Но текстовые рекомендации актуальны.",
    }
    return (EN if L == "EN" else RU).get(key, key)


# =========================
# UI: MAIN MENU
# =========================
def main_menu(chat_id: int) -> ReplyKeyboardMarkup:
    L = lang_of(chat_id)
    lang_btn = t(chat_id, "btn_lang_en") if L == "EN" else t(chat_id, "btn_lang_ru")

    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=t(chat_id, "btn_send"))],
            [KeyboardButton(text=t(chat_id, "btn_how"))],
            [KeyboardButton(text=t(chat_id, "btn_channel"))],
            [KeyboardButton(text=lang_btn)],
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
# MarkdownV2 helpers (for monospace blocks)
# =========================
def md2_escape_outside_code(s: str) -> str:
    # We avoid using MarkdownV2 for normal text; only for code blocks.
    # Still, keep a minimal escape if needed.
    to_escape = r"_*[]()~`>#+-=|{}.!"
    out = ""
    for ch in s:
        if ch in to_escape:
            out += "\\" + ch
        else:
            out += ch
    return out


def md2_codeblock(text: str, lang: str = "text") -> str:
    # Inside code block, only backticks can break it.
    safe = text.replace("```", "'''" )
    return f"```{lang}\n{safe}\n```"


# =========================
# ASCII ANIMATION (compact, dynamic, monospace)
# =========================
SPIN = ["|", "/", "-", "\\"]
PAT = ["░", "▒", "▓", "█"]

def ascii_frame(step: int, title: str) -> str:
    bar_w = 24
    fill = step % (bar_w + 1)
    # little shimmer effect
    shimmer = PAT[(step // 2) % len(PAT)]
    bar = "[" + (shimmer * fill) + (" " * (bar_w - fill)) + "]"
    # keep title short to prevent Telegram edits failing
    title = title[:28]
    return f"{SPIN[step % 4]} {title}\n{bar}"


async def safe_edit_code(msg: Message, content: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> Message:
    try:
        return await msg.edit_text(
            md2_codeblock(content, "text"),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup,
        )
    except Exception:
        return msg


async def animate_progress(anchor: Message, title: str, done_evt: asyncio.Event, cancel_markup: InlineKeyboardMarkup) -> Message:
    m = await anchor.answer(
        md2_codeblock(ascii_frame(0, title), "text"),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=cancel_markup,
    )
    step = 1
    while not done_evt.is_set():
        await asyncio.sleep(0.16)
        await safe_edit_code(m, ascii_frame(step, title), reply_markup=cancel_markup)
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
    return url.strip()


def pil_open_image(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img


def download_url_bytes(url: str, max_bytes: int, timeout_s: int) -> Optional[bytes]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (DesignReviewerBot/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
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
def merge_blocks_to_lines(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []
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
        lines.append({"text": text, "bbox": bbox, "conf": conf_avg})

    lines.sort(key=lambda z: (z["bbox"][1], z["bbox"][0]))
    return lines


def extract_ocr_lines(img: Image.Image) -> List[Dict[str, Any]]:
    data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    words: List[Dict[str, Any]] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if not txt:
            continue
        if conf != -1.0 and conf < 35:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append({
            "text": txt,
            "conf": conf,
            "bbox": [x, y, w, h],
            "line": int(data.get("line_num", [0]*n)[i]) if "line_num" in data else 0,
            "block": int(data.get("block_num", [0]*n)[i]) if "block_num" in data else 0,
            "par": int(data.get("par_num", [0]*n)[i]) if "par_num" in data else 0,
        })
    return merge_blocks_to_lines(words)


# =========================
# ANNOTATIONS
# =========================
def draw_annotations(img_bytes: bytes, lines: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> bytes:
    img = pil_open_image(img_bytes)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for k, iss in enumerate(issues, start=1):
        idx = iss.get("target_index", None)
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(lines):
            continue
        x, y, w, h = lines[idx]["bbox"]
        for t in range(3):
            draw.rectangle([x - t, y - t, x + w + t, y + h + t], outline=(0, 0, 0))

        label = str(k)
        lx = clamp(x - 6, 0, img.size[0]-1)
        ly = clamp(y - 28, 0, img.size[1]-1)
        bb = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = 6
        draw.rectangle([lx, ly, lx + tw + pad*2, ly + th + pad*2], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((lx + pad, ly + pad), label, fill=(0, 0, 0), font=font)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# =========================
# LLM + FALLBACK
# =========================
def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


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


def trim_ascii(s: str, max_lines: int = 18, max_width: int = 52) -> str:
    lines = s.splitlines()[:max_lines]
    lines = [ln[:max_width] for ln in lines]
    return "\n".join(lines).strip()


def heuristic_review(lines: List[Dict[str, Any]], chat_id: int, note: str = "") -> Dict[str, Any]:
    all_text = " ".join([l["text"] for l in lines]).lower()
    praise, ux, tx, issues = [], [], [], []

    if len(lines) >= 3:
        praise.append("There’s some structure on the screen — it doesn’t look like pure chaos.")

    if "error" in all_text or "ошиб" in all_text:
        tx.append("If there’s an error, make it concrete: what happened + what to do next.")

    ux.append("Check visual hierarchy: one primary focus, everything else supports it.")
    ux.append("If there’s too much text, compress into: title + 1–2 lines, then details if needed.")

    if lines:
        issues.append({
            "target_index": 0,
            "title": "Too generic",
            "problem": "Text feels vague — user may not understand what happens next.",
            "fix": "Rewrite as fact + next step: “We did X. Result Y. Next: go to Z.”",
            "severity": "medium",
        })

    if note:
        tx.append(note)

    what_i_see = "I see a UI screen with text elements and controls. Without LLM, I can only give general feedback."
    score = 6

    if lang_of(chat_id) == "RU":
        what_i_see = "Вижу экран UI с текстовыми элементами и контролами. Без LLM могу дать только общее ревью."
        score = 6

    return {
        "what_i_see": what_i_see,
        "score": score,
        "verdict": {"ux": ux, "text": tx, "praise": praise},
        "issues": issues,
        "concept_ascii": default_ascii_concept(),
    }


def normalize_llm_output(d: Dict[str, Any], lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["what_i_see"] = str(d.get("what_i_see", "")).strip()

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
    out["concept_ascii"] = trim_ascii(concept)

    return out


def call_llm_review(img_bytes: bytes, lines: List[Dict[str, Any]], chat_id: int) -> Dict[str, Any]:
    # Strong diagnostics + safe fallback
    if not LLM_ENABLED:
        return heuristic_review(lines, chat_id, note=t(chat_id, "llm_disabled"))
    if not OPENAI_API_KEY:
        return heuristic_review(lines, chat_id, note=t(chat_id, "llm_missing_key"))
    if OpenAI is None:
        return heuristic_review(lines, chat_id, note=t(chat_id, "llm_sdk_missing"))

    client = OpenAI(api_key=OPENAI_API_KEY)

    ocr_list = []
    for i, ln in enumerate(lines[:80]):
        ocr_list.append({"i": i, "text": ln["text"]})

    # All instructions in EN by default; RU mode asks it to answer in RU.
    if lang_of(chat_id) == "RU":
        lang_instr = "Write all fields in Russian."
    else:
        lang_instr = "Write all fields in English."

    prompt = f"""
You are a strict-but-fair senior product designer reviewing a UI screenshot.

Tone:
- Be blunt and specific, but no insults and no profanity.
- Praise with specifics when deserved.

Constraints:
- Fonts/palette: only GUESS font family vibe (e.g., "Inter-like", "SF Pro-like"). No sizes, no numeric colors.
- Always say what is wrong AND what to do instead.
- Score 0..10 (integer).
- Issues should map to OCR line indices when possible.

{lang_instr}

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

    try:
        # Responses API content types: input_text / input_image
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
            return heuristic_review(lines, chat_id, note="LLM returned non-JSON; fallback used.")
        return normalize_llm_output(parsed, lines)
    except Exception as e:
        # Print to logs (Railway)
        print("LLM ERROR:", repr(e))
        return heuristic_review(lines, chat_id, note=t(chat_id, "llm_error_prefix") + str(e))


# =========================
# FORMATTING (plain text, no HTML; monospace only for code blocks)
# =========================
def fmt_bullets(items: List[str], prefix: str = "- ") -> str:
    return "\n".join([prefix + x for x in items if x.strip()])


def format_what_i_see(r: Dict[str, Any], chat_id: int) -> str:
    s = (r.get("what_i_see") or "").strip()
    if not s:
        s = "I see a UI screen with text blocks and interface controls."
    return f"{t(chat_id, 'what_i_see')}\n{s}"


def format_verdict(r: Dict[str, Any], chat_id: int) -> str:
    score = r.get("score", 0)
    v = r.get("verdict", {}) if isinstance(r.get("verdict", {}), dict) else {}
    ux = v.get("ux", []) if isinstance(v.get("ux", []), list) else []
    tx = v.get("text", []) if isinstance(v.get("text", []), list) else []
    pr = v.get("praise", []) if isinstance(v.get("praise", []), list) else []

    parts = []
    parts.append(f"{t(chat_id, 'verdict_title')} {score}/10")

    if pr:
        parts.append("\n" + t(chat_id, "good") + "\n" + fmt_bullets(pr))
    if ux:
        parts.append("\n" + t(chat_id, "ux") + "\n" + fmt_bullets(ux))
    if tx:
        parts.append("\n" + t(chat_id, "text") + "\n" + fmt_bullets(tx))

    issues = r.get("issues", []) if isinstance(r.get("issues", []), list) else []
    if issues:
        lines_out = []
        for i, it in enumerate(issues[:12], start=1):
            title = (it.get("title") or "Issue").strip()
            sev = (it.get("severity") or "medium").strip().lower()
            problem = (it.get("problem") or "").strip()
            fix = (it.get("fix") or "").strip()
            lines_out.append(f"{i}) [{sev}] {title}\n   - Problem: {problem}\n   - Fix: {fix}")
        parts.append("\n" + t(chat_id, "callouts") + "\n" + "\n".join(lines_out))

    return "\n".join(parts).strip()


# =========================
# CORE PROCESSING (CANCEL + MENU POLICY)
# =========================
async def ocr_lines_async(img: Image.Image) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(extract_ocr_lines, img)


async def llm_review_async(img_bytes: bytes, lines: List[Dict[str, Any]], chat_id: int) -> Dict[str, Any]:
    return await asyncio.to_thread(call_llm_review, img_bytes, lines, chat_id)


async def process_image_review(anchor: Message, img_bytes: bytes, source_label: str = "SCREEN") -> None:
    chat_id = anchor.chat.id
    cancel_evt = asyncio.Event()
    active_reviews[chat_id] = cancel_evt

    done_evt = asyncio.Event()
    spinner_task = asyncio.create_task(
        animate_progress(anchor, f"REVIEW {source_label}", done_evt, cancel_keyboard())
    )

    try:
        img = pil_open_image(img_bytes)

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

        lines = await asyncio.wait_for(ocr_lines_async(img), timeout=OCR_TIMEOUT)

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

        review = await asyncio.wait_for(llm_review_async(img_bytes, lines, chat_id), timeout=LLM_TIMEOUT)

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

    except asyncio.CancelledError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(t(chat_id, "cancelled"), reply_markup=main_menu(chat_id))
        return

    except asyncio.TimeoutError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(t(chat_id, "timeout"), reply_markup=main_menu(chat_id))
        return

    except Exception as e:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(t(chat_id, "processing_failed") + str(e), reply_markup=main_menu(chat_id))
        return

    finally:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        active_reviews.pop(chat_id, None)

    # 1) What I see (no menu)
    await anchor.answer(format_what_i_see(review, chat_id))

    # 2) Verdict (no menu)
    await anchor.answer(format_verdict(review, chat_id))

    # 3) Annotated screenshot (no menu)
    try:
        annotated = await asyncio.to_thread(draw_annotations, img_bytes, lines, review.get("issues", []))
        await anchor.answer_photo(
            BufferedInputFile(annotated, filename="annotated.png"),
            caption=t(chat_id, "annotations_caption"),
        )
    except Exception:
        await anchor.answer(t(chat_id, "annot_fail"))

    # Short retro intermission (keeps style, not huge)
    done2 = asyncio.Event()
    spinner2 = asyncio.create_task(animate_progress(anchor, "CONCEPT", done2, cancel_keyboard()))
    await asyncio.sleep(0.75)
    done2.set()
    try:
        await spinner2
    except Exception:
        pass

    # 4) Concept (monospace)
    concept = review.get("concept_ascii") or default_ascii_concept()
    concept = trim_ascii(str(concept))

    await anchor.answer(
        t(chat_id, "concept_title") + "\n" + md2_codeblock(concept, "text"),
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    # menu only at the end
    await anchor.answer(t(chat_id, "done"), reply_markup=main_menu(chat_id))


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

        thumb = o.get("thumbnail_url") or ""
        if not thumb:
            raise RuntimeError("No thumbnail_url from Figma. Make sure the file is public.")

        if cancel_evt.is_set():
            raise asyncio.CancelledError()

        img_bytes = await asyncio.wait_for(
            asyncio.to_thread(download_url_bytes, thumb, MAX_PREVIEW_BYTES, FIGMA_DOWNLOAD_TIMEOUT),
            timeout=FIGMA_DOWNLOAD_TIMEOUT
        )
        if not img_bytes:
            raise RuntimeError("Failed to download preview (too large / blocked).")

    except asyncio.CancelledError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(t(chat_id, "cancelled"), reply_markup=main_menu(chat_id))
        return
    except asyncio.TimeoutError:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(t(chat_id, "fetch_timeout"), reply_markup=main_menu(chat_id))
        return
    except Exception as e:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        await anchor.answer(t(chat_id, "fetch_failed") + str(e), reply_markup=main_menu(chat_id))
        return
    finally:
        done_evt.set()
        try:
            await spinner_task
        except Exception:
            pass
        active_reviews.pop(chat_id, None)

    # Show preview so user knows we fetched correct node
    await anchor.answer_photo(
        BufferedInputFile(img_bytes, filename="figma_preview.png"),
        caption=t(chat_id, "figma_preview_caption"),
    )

    await process_image_review(anchor, img_bytes, source_label="FIGMA")


# =========================
# HANDLERS
# =========================
@router.message(Command("start"))
async def on_start(m: Message):
    chat_id = m.chat.id
    chat_lang.setdefault(chat_id, "EN")
    await m.answer(
        f"{t(chat_id, 'start_title')}\n\n{t(chat_id, 'start_body')}",
        reply_markup=main_menu(chat_id)
    )


@router.callback_query(F.data == BTN_CANCEL)
async def on_cancel(cb: CallbackQuery):
    chat_id = cb.message.chat.id if cb.message else None
    if chat_id is not None:
        evt = active_reviews.get(chat_id)
        if evt:
            evt.set()
    try:
        if cb.message:
            await cb.message.edit_text("Cancelling…")
    except Exception:
        pass
    await cb.answer("Cancelled")


@router.message(F.text)
async def on_text(m: Message):
    chat_id = m.chat.id
    txt = (m.text or "").strip()

    # Language toggle button
    if txt == t(chat_id, "btn_lang_en") or txt == t(chat_id, "btn_lang_ru"):
        chat_lang[chat_id] = "RU" if lang_of(chat_id) == "EN" else "EN"
        await m.answer(
            "OK." if lang_of(chat_id) == "EN" else "Ок.",
            reply_markup=main_menu(chat_id)
        )
        return

    # Menu buttons
    if txt == t(chat_id, "btn_how"):
        await m.answer(t(chat_id, "how"))
        return

    if txt == t(chat_id, "btn_send"):
        await m.answer(t(chat_id, "send_hint"))
        return

    if txt == t(chat_id, "btn_channel"):
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Open @prodooktovy", url="https://t.me/prodooktovy")]]
        )
        await m.answer(t(chat_id, "channel"), reply_markup=kb)
        return

    # Figma link
    if looks_like_figma_url(txt):
        await process_figma_link(m, txt)
        return

    # Other URL
    if is_probably_url(txt):
        await m.answer(t(chat_id, "figma_bad_link"), reply_markup=main_menu(chat_id))
        return

    # Random text
    await m.answer(t(chat_id, "text_alone"))


@router.message(F.content_type == ContentType.PHOTO)
async def on_photo(m: Message, bot: Bot):
    chat_id = m.chat.id
    ph = m.photo[-1]
    file = await bot.get_file(ph.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    img_bytes = buf.getvalue()
    if len(img_bytes) > MAX_IMAGE_BYTES:
        await m.answer(t(chat_id, "img_too_large"), reply_markup=main_menu(chat_id))
        return
    await process_image_review(m, img_bytes, source_label="SCREEN")


@router.message(F.content_type == ContentType.DOCUMENT)
async def on_document(m: Message, bot: Bot):
    chat_id = m.chat.id
    doc = m.document
    if not doc:
        return
    mime = (doc.mime_type or "").lower()
    name = (doc.file_name or "").lower()
    if not (mime.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".webp"))):
        await m.answer(t(chat_id, "doc_need_img"), reply_markup=main_menu(chat_id))
        return

    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    img_bytes = buf.getvalue()
    if len(img_bytes) > MAX_IMAGE_BYTES:
        await m.answer(t(chat_id, "img_too_large"), reply_markup=main_menu(chat_id))
        return
    await process_image_review(m, img_bytes, source_label="SCREEN")


# =========================
# MAIN
# =========================
async def main():
    # Keep default parse_mode None: we only use MarkdownV2 explicitly when needed for code blocks.
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=None)
    )
    dp = Dispatcher()
    dp.include_router(router)

    try:
        await bot.set_my_commands([BotCommand(command="start", description="Start")])
    except Exception:
        pass

    print(f"✅ Starting… OCR_LANG={OCR_LANG}, LLM_ENABLED={LLM_ENABLED}, model={LLM_MODEL}, has_key={bool(OPENAI_API_KEY)}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())