#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Design Reviewer Bot (Telegram)

Flow (per screenshot / public Figma frame link):
1) Text: what I see
2) Text: verdict + recommendations (UX + copy) + score /10
3) Image: annotated screenshot (boxes/arrows/labels)
4) ASCII retro progress animation
5) Image: concept draft (resized to original screenshot size)

Notes:
- Emojis: mostly monochrome/neutral
- No technical measurements (no exact font sizes / color values). Font family guesses only.
- Avoids leaking JSON/arrays in user-facing messages.
- Figma link support uses Figma oEmbed thumbnail (works only if file is public).
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import html as pyhtml

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

import pytesseract
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import Command
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)

from openai import OpenAI


# -----------------------------
# Env / Config
# -----------------------------

def load_env() -> None:
    # Robust .env loading (works in scripts, REPL, containers)
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)
    else:
        load_dotenv(override=False)


load_env()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

OCR_LANG = os.getenv("OCR_LANG", "rus+eng").strip()
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

# Optional: in macOS you might need to set this:
# export TESSERACT_CMD="/opt/homebrew/bin/tesseract"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in .env or environment")
if LLM_ENABLED and not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env or environment if LLM_ENABLED=true")

client = OpenAI(api_key=OPENAI_API_KEY) if LLM_ENABLED else None


# -----------------------------
# UI (Keyboards)
# -----------------------------

BTN_SUBMIT = "Submit for review"
BTN_HOW = "How does it work?"
BTN_PING = "Ping"

MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_SUBMIT)],
        [KeyboardButton(text=BTN_HOW), KeyboardButton(text=BTN_PING)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Send a screenshot or a public Figma frame link…",
)


# -----------------------------
# ASCII retro progress
# -----------------------------

ASCII_FRAMES = [
    r"""[=         ]""",
    r"""[==        ]""",
    r"""[===       ]""",
    r"""[====      ]""",
    r"""[=====     ]""",
    r"""[======    ]""",
    r"""[=======   ]""",
    r"""[========  ]""",
    r"""[========= ]""",
    r"""[==========]""",
]

def ascii_line(i: int, label: str) -> str:
    i = max(0, min(i, len(ASCII_FRAMES) - 1))
    return f"{label}\n{ASCII_FRAMES[i]}"

async def send_ascii_progress(chat_msg: Message, label: str, delay: float = 0.12) -> None:
    # Do NOT edit messages (Telegram sometimes refuses edits).
    # Instead send a short series of new messages.
    for i in range(0, len(ASCII_FRAMES), 2):
        await chat_msg.answer(f"<code>{pyhtml.escape(ascii_line(i, label))}</code>")
        await asyncio.sleep(delay)


# -----------------------------
# Helpers: detect input type
# -----------------------------

FIGMA_URL_RE = re.compile(r"https?://www\.figma\.com/(design|file)/[^ ]+", re.IGNORECASE)

def is_figma_url(text: str) -> bool:
    return bool(FIGMA_URL_RE.search(text or ""))

def extract_node_id(url: str) -> Optional[str]:
    # Accept node-id=1-2 or node-id=2%3A88 etc.
    m = re.search(r"node-id=([^&]+)", url)
    if not m:
        return None
    raw = m.group(1)
    return raw.replace("%3A", ":")  # figma sometimes uses 2:88
    # keep "1-2" as-is too


# -----------------------------
# Figma fetch via oEmbed
# -----------------------------

async def fetch_figma_thumbnail(url: str) -> Optional[bytes]:
    """
    Uses Figma oEmbed endpoint.
    Works only if file is public and Figma provides a thumbnail.
    """
    oembed = f"https://www.figma.com/api/oembed?url={pyhtml.escape(url, quote=True)}"
    async with httpx.AsyncClient(timeout=20) as hc:
        r = await hc.get(oembed)
        if r.status_code != 200:
            return None
        data = r.json()
        thumb = data.get("thumbnail_url")
        if not thumb:
            return None
        img = await hc.get(thumb)
        if img.status_code != 200:
            return None
        return img.content


# -----------------------------
# OCR
# -----------------------------

@dataclass
class OCRBox:
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    conf: float

def ocr_boxes(pil_img: Image.Image, lang: str) -> List[OCRBox]:
    # Use tesseract TSV output for bboxes
    # Try to reduce noise: keep only confident tokens/words.
    tsv = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
    n = len(tsv.get("text", []))
    out: List[OCRBox] = []
    for i in range(n):
        txt = (tsv["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(tsv["conf"][i])
        except Exception:
            conf = -1.0
        # conservative threshold
        if conf < 45:
            continue
        x, y, w, h = int(tsv["left"][i]), int(tsv["top"][i]), int(tsv["width"][i]), int(tsv["height"][i])
        out.append(OCRBox(text=txt, bbox=(x, y, w, h), conf=conf))
    return out

def ocr_text_snippet(boxes: List[OCRBox], max_tokens: int = 120) -> str:
    # Join text in reading order approximation (tsv order is usually left-to-right-ish).
    tokens = [b.text for b in boxes]
    return " ".join(tokens[:max_tokens])


# -----------------------------
# LLM: structured output + robust parsing
# -----------------------------

REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "what_i_see": {"type": "string"},
        "score_10": {"type": "integer", "minimum": 1, "maximum": 10},
        "verdict": {"type": "string"},
        "praise": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "why_bad": {"type": "string"},
                    "fix": {"type": "string"},
                    "category": {"type": "string", "enum": ["ux", "copy", "visual", "accessibility", "consistency"]},
                    # For annotation targeting:
                    "target_text": {"type": "string"},  # optional: short snippet seen on screen
                    "region": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "x": {"type": "number", "minimum": 0, "maximum": 1},
                            "y": {"type": "number", "minimum": 0, "maximum": 1},
                            "w": {"type": "number", "minimum": 0, "maximum": 1},
                            "h": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["x", "y", "w", "h"]
                    },
                },
                "required": ["title", "why_bad", "fix", "category"]
            },
            "maxItems": 10
        },
        "font_family_guess": {"type": "string"},
        "concept_prompt": {"type": "string"},
    },
    "required": ["what_i_see", "score_10", "verdict", "praise", "issues", "font_family_guess", "concept_prompt"],
}

def _extract_json_text(resp: Any) -> str:
    """
    openai responses API can return text in different shapes.
    We try to get the final text.
    """
    # New SDK often gives resp.output_text
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text.strip()
    # Fallback: stringify and try to find JSON object
    s = str(resp)
    return s

def _safe_json_loads(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s:
        return None
    # Try direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def llm_review(
    pil_img: Image.Image,
    img_b64_data_url: str,
    ocr_hint: str,
) -> Dict[str, Any]:
    assert client is not None

    system = (
        "You are a strict senior UI/UX design reviewer.\n"
        "You must be honest, sometimes tough, but no profanity.\n"
        "You can praise if deserved and explain what is good.\n"
        "No technical measurement outputs: no exact font sizes, no color hex values.\n"
        "You may guess font family.\n"
        "Output MUST be valid JSON matching the provided schema.\n"
        "Do not wrap text in arrays like ['...'].\n"
        "Do not include keys like {'description': ...} in user-facing strings.\n"
    )

    user = (
        "Review this UI screenshot. Provide:\n"
        "1) what_i_see: short description of the screen and user scenario.\n"
        "2) verdict: a single message that mixes UX + copy recommendations.\n"
        "3) score_10: overall score 1..10.\n"
        "4) praise: 1..3 concrete praises if any.\n"
        "5) issues: up to 8 key issues (prioritized). Each issue must include: title, why_bad, fix, category.\n"
        "   For annotations: optionally include target_text (short snippet that exists on screen) and/or region (x,y,w,h normalized).\n"
        "6) font_family_guess: one short guess like 'Inter', 'SF Pro', 'Roboto', 'Helvetica', etc.\n"
        "7) concept_prompt: a prompt to generate a concept redesign image (same language as the UI, keep meaning, improve hierarchy).\n\n"
        f"OCR hint (may be imperfect): {ocr_hint}\n"
    )

    # We’ll use Responses API with json_schema formatting
    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": img_b64_data_url},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "design_reviewer_v1",
                "schema": REVIEW_SCHEMA,
            }
        },
        max_output_tokens=900,
    )

    txt = _extract_json_text(resp)
    data = _safe_json_loads(txt)
    if not data:
        # Last resort: still provide a controlled fallback
        raise ValueError("LLM returned invalid JSON")
    return data


# -----------------------------
# Annotation image rendering
# -----------------------------

def _load_font(size: int = 18) -> ImageFont.FreeTypeFont:
    # Use a built-in fallback if system fonts are not available
    # Try common fonts first
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # mac
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        # linux
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def _best_match_box(boxes: List[OCRBox], target_text: str) -> Optional[OCRBox]:
    if not target_text:
        return None
    t = target_text.strip().lower()
    if not t:
        return None

    # Prefer boxes that contain the token (or part of it)
    best = None
    best_score = 0.0

    for b in boxes:
        s = b.text.lower()
        score = 0.0
        if s == t:
            score = 1.0
        elif t in s or s in t:
            score = 0.7
        else:
            # weak fuzzy: common prefix length
            pref = 0
            for a, c in zip(s, t):
                if a == c:
                    pref += 1
                else:
                    break
            score = pref / max(1, max(len(s), len(t)))
        score = score * (0.5 + min(0.5, b.conf / 100.0))
        if score > best_score:
            best_score = score
            best = b

    if best_score < 0.35:
        return None
    return best

def render_annotated(
    base_img: Image.Image,
    issues: List[Dict[str, Any]],
    boxes: List[OCRBox],
) -> Image.Image:
    img = base_img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = _load_font(18)
    font_small = _load_font(16)

    W, H = img.size
    pad = max(3, int(min(W, H) * 0.003))

    # Retro-ish monochrome annotation style: black outline + white fill label
    def draw_label(x: int, y: int, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # background
        tw, th = draw.textbbox((0, 0), text, font=font_small)[2:]
        bx1, by1 = x, y
        bx2, by2 = x + tw + 10, y + th + 8
        draw.rectangle([bx1, by1, bx2, by2], fill=(255, 255, 255, 235), outline=(0, 0, 0, 255), width=2)
        draw.text((x + 5, y + 4), text, font=font_small, fill=(0, 0, 0, 255))

    # Limit to avoid clutter
    issues_to_draw = issues[:8]

    for idx, it in enumerate(issues_to_draw, start=1):
        title = (it.get("title") or "").strip()
        target_text = (it.get("target_text") or "").strip()

        # Determine box
        box = _best_match_box(boxes, target_text) if target_text else None

        if box:
            x, y, w, h = box.bbox
        else:
            reg = it.get("region") or None
            if isinstance(reg, dict):
                x = int((reg.get("x", 0.1)) * W)
                y = int((reg.get("y", 0.1)) * H)
                w = int((reg.get("w", 0.2)) * W)
                h = int((reg.get("h", 0.08)) * H)
            else:
                # fallback: top-left-ish
                x, y, w, h = int(0.08 * W), int(0.12 * H) + idx * 10, int(0.2 * W), int(0.08 * H)

        # Clamp
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(10, min(w, W - x))
        h = max(10, min(h, H - y))

        # Draw rectangle (monochrome)
        draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], outline=(0, 0, 0, 255), width=4)

        # Number bubble
        bubble = f"{idx}"
        bw, bh = draw.textbbox((0, 0), bubble, font=font)[2:]
        bx1, by1 = x - pad, max(0, y - bh - 14)
        bx2, by2 = bx1 + bw + 14, by1 + bh + 10
        draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, 255), outline=(0, 0, 0, 255))
        draw.text((bx1 + 7, by1 + 5), bubble, font=font, fill=(255, 255, 255, 255))

        # Short label near box
        short = title
        if len(short) > 42:
            short = short[:39].rstrip() + "…"
        draw_label(min(W - 220, x + w + 10), min(H - 40, y), short)

    return img.convert("RGB")


# -----------------------------
# Concept image generation
# -----------------------------

def generate_concept_image(concept_prompt: str, target_size: Tuple[int, int]) -> Image.Image:
    """
    Generates a concept image using Images API (gpt-image-1) and resizes to target_size.
    If your account/model availability differs, change model accordingly.
    """
    assert client is not None

    W, H = target_size

    # Choose a supported size closest to aspect ratio
    # Common supported sizes: 1024x1024, 1536x1024, 1024x1536
    if W >= H:
        gen_size = "1536x1024" if (W / max(1, H)) > 1.15 else "1024x1024"
    else:
        gen_size = "1024x1536" if (H / max(1, W)) > 1.15 else "1024x1024"

    prompt = (
        "Create a CONCEPT UI mock based on the following guidance.\n"
        "Important: this is a draft direction, not pixel-perfect.\n"
        "Keep the same language as the original UI.\n"
        "Improve hierarchy, clarity, spacing, and CTA focus.\n"
        "No decorative nonsense.\n\n"
        f"{concept_prompt}\n"
    )

    img_resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=gen_size,
    )

    # openai images response typically contains base64 data
    b64 = img_resp.data[0].b64_json
    raw = base64.b64decode(b64)
    im = Image.open(BytesIO(raw)).convert("RGB")

    # Resize to exact screenshot size (as you requested)
    im = im.resize((W, H), Image.LANCZOS)
    return im


# -----------------------------
# Telegram: formatting helpers
# -----------------------------

def safe_text(s: str) -> str:
    # Avoid accidental HTML entities in Telegram
    return pyhtml.escape((s or "").strip())

def format_message_what_i_see(what: str, font_guess: str) -> str:
    return (
        "WHAT I SEE\n"
        f"{safe_text(what)}\n\n"
        f"Font family (guess): {safe_text(font_guess)}"
    )

def format_message_verdict(score_10: int, verdict: str, praise: List[str], issues: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    parts.append(f"VERDICT — {int(score_10)}/10\n")
    parts.append(safe_text(verdict).strip())

    # Praise (keep short)
    praise_clean = [p.strip() for p in (praise or []) if p and p.strip()]
    if praise_clean:
        parts.append("\nWHAT'S GOOD")
        for p in praise_clean[:3]:
            parts.append(f"• {safe_text(p)}")

    # Issues (summarized)
    issues_clean = issues or []
    if issues_clean:
        parts.append("\nMAIN ISSUES (PRIORITY)")
        for i, it in enumerate(issues_clean[:6], start=1):
            title = safe_text(it.get("title", "Issue"))
            why = safe_text(it.get("why_bad", "")).strip()
            fix = safe_text(it.get("fix", "")).strip()
            # Keep concise, but actionable
            parts.append(f"{i}. {title}")
            if why:
                parts.append(f"   Why: {why}")
            if fix:
                parts.append(f"   Fix: {fix}")

    return "\n".join(parts).strip()


# -----------------------------
# Router / Handlers
# -----------------------------

router = Router()


@router.message(Command("start"))
async def cmd_start(m: Message) -> None:
    txt = (
        "Design Reviewer online.\n"
        "Send a screenshot or a public Figma frame link.\n"
        "I'll be honest: I will praise what works and call out what doesn't."
    )
    await m.answer(txt, reply_markup=MAIN_KB)


@router.message(Command("ping"))
async def cmd_ping(m: Message) -> None:
    await m.answer("pong", reply_markup=MAIN_KB)


@router.message(F.text == BTN_HOW)
async def how_it_works(m: Message) -> None:
    txt = (
        "HOW IT WORKS\n"
        "1) Send a screenshot or a public Figma frame link\n"
        "2) Get: what I see, verdict + fixes, annotated screenshot, concept draft\n"
        "That's it."
    )
    await m.answer(txt, reply_markup=MAIN_KB)


@router.message(F.text == BTN_PING)
async def ping_button(m: Message) -> None:
    await cmd_ping(m)


@router.message(F.text == BTN_SUBMIT)
async def submit_button(m: Message) -> None:
    await m.answer(
        "Send a screenshot or a public Figma frame link.",
        reply_markup=ReplyKeyboardRemove(),
    )
    await m.answer("Menu is still available anytime with /start.", reply_markup=MAIN_KB)


async def download_telegram_photo(m: Message) -> Optional[bytes]:
    if not m.photo:
        return None
    photo = m.photo[-1]
    file = await m.bot.get_file(photo.file_id)
    buf = BytesIO()
    await m.bot.download_file(file.file_path, destination=buf)
    return buf.getvalue()


def to_data_url_png(img_bytes: bytes) -> Tuple[str, Image.Image]:
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    out = BytesIO()
    pil.save(out, format="PNG")
    b = out.getvalue()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}", pil


def img_bytes_from_pil(pil: Image.Image, fmt: str = "PNG") -> bytes:
    bio = BytesIO()
    pil.save(bio, format=fmt)
    return bio.getvalue()


def cache_key_for_request(img_bytes: bytes, source_tag: str) -> str:
    h = hashlib.sha256(img_bytes).hexdigest()[:16]
    return f"{source_tag}:{h}:{int(time.time()*1000)}"  # include time to avoid reusing wrong previews


@router.message(F.content_type == ContentType.PHOTO)
async def handle_photo(m: Message) -> None:
    try:
        raw = await download_telegram_photo(m)
        if not raw:
            await m.answer("I didn't get the image. Try again.")
            return

        await process_image_request(m, raw, source_tag="tg_photo")

    except Exception as e:
        await m.answer(f"Processing crashed: {safe_text(str(e))}")


@router.message(F.text & F.text.len() > 0)
async def handle_text(m: Message) -> None:
    text = (m.text or "").strip()

    if is_figma_url(text):
        await m.answer("Got it. Trying to fetch a preview from Figma…", reply_markup=MAIN_KB)
        node_id = extract_node_id(text)
        thumb = None

        try:
            thumb = await fetch_figma_thumbnail(text)
        except Exception:
            thumb = None

        if not thumb:
            await m.answer(
                "I couldn't fetch a preview.\n"
                "Make sure the Figma file is public and the link is correct.",
                reply_markup=MAIN_KB,
            )
            return

        # Show preview image (as requested)
        await m.answer_photo(
            photo=thumb,
            caption=("Figma preview" + (f" (node-id={node_id})" if node_id else "")),
        )

        # Then process like a normal screenshot
        await process_image_request(m, thumb, source_tag=f"figma:{node_id or 'no_node'}")
        return

    # Default fallback
    if text in ("/start", "/ping"):
        return

    await m.answer(
        "I can review:\n"
        "• screenshots (images)\n"
        "• public Figma frame links\n\n"
        "Use the menu or just send a screenshot/link.",
        reply_markup=MAIN_KB,
    )


async def process_image_request(m: Message, img_bytes: bytes, source_tag: str) -> None:
    # New request — no cross-request cache.
    # (This avoids the “every link returns the same thing” bug.)
    req_id = cache_key_for_request(img_bytes, source_tag)

    # Step 0: initial progress (retro)
    await m.answer(f"<code>{pyhtml.escape(ascii_line(0, 'LOOKING…'))}</code>")

    data_url, pil = to_data_url_png(img_bytes)

    # OCR first (helps targeting + reduces “boxes on empty areas”)
    try:
        boxes = ocr_boxes(pil, OCR_LANG)
        ocr_hint = ocr_text_snippet(boxes)
    except Exception:
        boxes = []
        ocr_hint = ""

    if not LLM_ENABLED:
        await m.answer("LLM is disabled (LLM_ENABLED=false). Enable it to get reviews.")
        return

    # Step 1: LLM review (structured)
    await m.answer(f"<code>{pyhtml.escape(ascii_line(3, 'READING…'))}</code>")
    review = llm_review(pil_img=pil, img_b64_data_url=data_url, ocr_hint=ocr_hint)

    what_i_see = review.get("what_i_see", "").strip()
    score_10 = int(review.get("score_10", 6))
    verdict = review.get("verdict", "").strip()
    praise = review.get("praise", []) or []
    issues = review.get("issues", []) or []
    font_guess = review.get("font_family_guess", "").strip()
    concept_prompt = review.get("concept_prompt", "").strip()

    # Message 1: What I see
    await m.answer(format_message_what_i_see(what_i_see, font_guess), reply_markup=MAIN_KB)

    # Message 2: Verdict & recommendations (UX + copy in one)
    await m.answer(format_message_verdict(score_10, verdict, praise, issues), reply_markup=MAIN_KB)

    # Message 3: Annotated screenshot
    await m.answer(f"<code>{pyhtml.escape(ascii_line(6, 'MARKING…'))}</code>")
    annotated = render_annotated(base_img=pil, issues=issues, boxes=boxes)
    ann_bytes = img_bytes_from_pil(annotated, fmt="PNG")

    # Short legend in caption (avoid clutter)
    legend_lines = []
    for i, it in enumerate(issues[:8], start=1):
        t = (it.get("title") or "").strip()
        if t:
            if len(t) > 60:
                t = t[:57].rstrip() + "…"
            legend_lines.append(f"{i}. {t}")
    caption = "Annotations\n" + ("\n".join(legend_lines) if legend_lines else "Key areas marked.")
    await m.answer_photo(photo=ann_bytes, caption=caption)

    # ASCII progress between 3 and 4 (as requested)
    await send_ascii_progress(m, label="DRAFTING CONCEPT…", delay=0.10)

    # Message 4: Concept (image, resized to original screenshot size)
    await m.answer(f"<code>{pyhtml.escape(ascii_line(9, 'RENDERING…'))}</code>")

    # Generate concept and force exact screenshot size
    concept_img = generate_concept_image(concept_prompt, target_size=pil.size)
    concept_bytes = img_bytes_from_pil(concept_img, fmt="PNG")

    await m.answer_photo(
        photo=concept_bytes,
        caption="Concept draft. Not pixel-perfect. One possible direction.",
    )


# -----------------------------
# Main
# -----------------------------

async def main() -> None:
    dp = Dispatcher()
    dp.include_router(router)

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    print(f"OK: starting bot… OCR_LANG={OCR_LANG}, LLM_ENABLED={LLM_ENABLED}, model={LLM_MODEL}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())