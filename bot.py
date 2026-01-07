# bot.py
# Design Reviewer Telegram bot (Aiogram 3.7+)
# - Accepts screenshots (photos) and public Figma links (via oEmbed thumbnail)
# - Shows compact retro ASCII progress animation (loops until done)
# - Sends outputs:
#   1) What I see
#   2) Verdict + recommendations
#   3) ASCII wireframe concept (monospace, width-limited to avoid mobile wrapping)
#   4) References block (meaning-based; patterns + example products + Pinterest/Dribbble links; no images)
# - EN by default, RU toggle
# - Menu shown only after /start and at the end of each review
# - Cancel button works during processing
# - Annotated screenshot is DISABLED
# - Admin-only Stats button + /stats (admin = @uxd_ink)

import asyncio
import base64
import io
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from PIL import Image

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
)
from aiogram.client.default import DefaultBotProperties

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

# Admin username (without @)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "uxd_ink").strip().lstrip("@").lower()

# SQLite path (Railway: can set to /data/bot_stats.db if you mount a volume)
SQLITE_PATH = os.getenv("SQLITE_PATH", "bot_stats.db").strip()

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in environment (Railway Variables or local export).")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ----------------------------
# State (in-memory)
# ----------------------------

LANG: Dict[int, str] = {}  # user_id -> "en" / "ru"
CANCEL_FLAGS: Dict[int, bool] = {}  # user_id -> True when cancelled
RUNNING_TASK: Dict[int, asyncio.Task] = {}  # user_id -> running processing task

# Remember last known usernames for UI decisions (admin stats button)
USERNAMES: Dict[int, str] = {}  # user_id -> username (lower)


# ----------------------------
# DB (SQLite, zero deps)
# ----------------------------

_db_lock = asyncio.Lock()
_db = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
_db.execute(
    """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  username TEXT,
  lang TEXT,
  event TEXT NOT NULL,
  input_type TEXT,
  duration_ms INTEGER,
  score INTEGER,
  error TEXT
)
"""
)
_db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
_db.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")
_db.commit()


def _now_ts() -> int:
    return int(time.time())


async def log_event(
    *,
    user_id: int,
    username: str,
    lang: str,
    event: str,
    input_type: Optional[str] = None,
    duration_ms: Optional[int] = None,
    score: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    # Keep it robust: never crash bot because of analytics
    try:
        async with _db_lock:
            _db.execute(
                """
                INSERT INTO events (ts, user_id, username, lang, event, input_type, duration_ms, score, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_ts(),
                    int(user_id),
                    (username or "").strip()[:64],
                    (lang or "en").strip()[:8],
                    (event or "").strip()[:64],
                    (input_type or None),
                    int(duration_ms) if duration_ms is not None else None,
                    int(score) if score is not None else None,
                    (error or None),
                ),
            )
            _db.commit()
    except Exception:
        pass


async def get_stats_summary() -> Dict[str, Any]:
    async with _db_lock:
        now = _now_ts()
        day = now - 24 * 3600
        week = now - 7 * 24 * 3600

        def q(sql: str, params=()):
            cur = _db.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else 0

        total_users = q("SELECT COUNT(DISTINCT user_id) FROM events")
        total_reviews = q("SELECT COUNT(*) FROM events WHERE event='review_done'")
        reviews_24h = q("SELECT COUNT(*) FROM events WHERE event='review_done' AND ts>=?", (day,))
        reviews_7d = q("SELECT COUNT(*) FROM events WHERE event='review_done' AND ts>=?", (week,))
        cancels_7d = q("SELECT COUNT(*) FROM events WHERE event='review_cancelled' AND ts>=?", (week,))
        fails_7d = q("SELECT COUNT(*) FROM events WHERE event='review_failed' AND ts>=?", (week,))
        photo_7d = q("SELECT COUNT(*) FROM events WHERE event='review_done' AND input_type='photo' AND ts>=?", (week,))
        figma_7d = q("SELECT COUNT(*) FROM events WHERE event='review_done' AND input_type='figma' AND ts>=?", (week,))

        # Top errors (last 7d)
        cur = _db.execute(
            """
            SELECT error, COUNT(*) as c
            FROM events
            WHERE event='review_failed' AND ts>=? AND error IS NOT NULL
            GROUP BY error
            ORDER BY c DESC
            LIMIT 5
            """,
            (week,),
        )
        top_errors = [(r[0], r[1]) for r in cur.fetchall()]

    return {
        "total_users": total_users,
        "total_reviews": total_reviews,
        "reviews_24h": reviews_24h,
        "reviews_7d": reviews_7d,
        "cancels_7d": cancels_7d,
        "fails_7d": fails_7d,
        "photo_7d": photo_7d,
        "figma_7d": figma_7d,
        "top_errors": top_errors,
    }


# ----------------------------
# Helpers: i18n + HTML escaping
# ----------------------------

def lang_for(uid: int) -> str:
    return LANG.get(uid, "en")


def set_lang(uid: int, value: str) -> None:
    LANG[uid] = "ru" if value == "ru" else "en"


def _lang_is_ru(lang: str) -> bool:
    return (lang or "en").lower().startswith("ru")


def escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def is_admin_message(m: Message) -> bool:
    u = (m.from_user.username or "").strip().lower()
    return u == ADMIN_USERNAME


def is_admin_uid(uid: int) -> bool:
    u = USERNAMES.get(uid, "").lower()
    return u == ADMIN_USERNAME


TXT = {
    "en": {
        "title": "Design Reviewer",
        "start": (
            "I'm your design review partner.\n\n"
            "Send me:\n"
            "• screenshots (images)\n"
            "• Figma frame links (public files)\n\n"
            "Tap “Submit for review” or just send a screenshot/link."
        ),
        "how": (
            "How it works:\n"
            "1) Send a screenshot or a public Figma frame link\n"
            "2) I analyze what’s on the screen\n"
            "3) I give a tough-but-fair review + a wireframe concept\n"
            "4) I give references (patterns + example products + links)"
        ),
        "need_input": "Send a screenshot or a public Figma frame link.",
        "cancelled": "Cancelled.",
        "busy": "I’m already reviewing something. Hit Cancel, or wait for the result.",
        "no_llm": "LLM is disabled (no OPENAI_API_KEY). Add it to Railway Variables.",
        "llm_fail": "LLM error. Try again (or send a clearer / larger screenshot).",
        "refs_fail": "Couldn’t build references this time. Try again.",
        "not_image": "I need a screenshot image or a Figma link.",
        "figma_fetch_fail": "Couldn’t fetch a preview from that Figma link. Make sure the file is public.",
        "whatisee": "What I see",
        "verdict": "Verdict & recommendations",
        "concept": "Wireframe concept (ASCII)",
        "refs": "References (meaning-based)",
        "score": "Score",
        "channel_msg": "Channel:",
        "menu_submit": "Submit for review",
        "menu_how": "How it works?",
        "menu_lang": "Language: EN/RU",
        "menu_channel": "@prodooktovy",
        "menu_stats": "Stats",
        "btn_cancel": "Cancel",
        "progress_title": "Review in progress",
        "progress_wire": "Drafting wireframe",
        "progress_refs": "Finding references",
        "preview": "Preview",
        "refs_sub": "Patterns → examples → links",
        "label_why": "Why:",
        "label_examples": "Examples:",
        "label_look": "Look for:",
        "label_links": "Links:",
    },
    "ru": {
        "title": "Design Reviewer",
        "start": (
            "Я твой партнёр по дизайн-ревью.\n\n"
            "Принимаю:\n"
            "• скриншоты (картинки)\n"
            "• ссылки на фреймы Figma (если файл публичный)\n\n"
            "Жми «Закинуть на ревью» или просто отправь скрин/ссылку."
        ),
        "how": (
            "Как это работает:\n"
            "1) Отправь скриншот или публичную ссылку на Figma фрейм\n"
            "2) Я опишу, что вижу на экране\n"
            "3) Дам честный разбор + черновой вайрфрейм\n"
            "4) Дам референсы: паттерны + примеры продуктов + ссылки"
        ),
        "need_input": "Отправь скриншот или публичную ссылку на Figma фрейм.",
        "cancelled": "Отменено.",
        "busy": "Я уже делаю ревью. Нажми Cancel или дождись результата.",
        "no_llm": "LLM отключён (нет OPENAI_API_KEY). Добавь его в Railway Variables.",
        "llm_fail": "Ошибка LLM. Попробуй ещё раз (или пришли скрин крупнее/четче).",
        "refs_fail": "Не получилось собрать референсы. Попробуй ещё раз.",
        "not_image": "Мне нужен скриншот или ссылка на Figma.",
        "figma_fetch_fail": "Не смог скачать превью по ссылке Figma. Убедись, что файл публичный.",
        "whatisee": "Что я вижу",
        "verdict": "Вердикт и рекомендации",
        "concept": "Wireframe-концепт (ASCII)",
        "refs": "Референсы по смыслу",
        "score": "Оценка",
        "channel_msg": "Канал:",
        "menu_submit": "Закинуть на ревью",
        "menu_how": "Как это работает?",
        "menu_lang": "Язык: EN/RU",
        "menu_channel": "@prodooktovy",
        "menu_stats": "Статистика",
        "btn_cancel": "Cancel",
        "progress_title": "Ревью в процессе",
        "progress_wire": "Собираю wireframe",
        "progress_refs": "Подбираю референсы",
        "preview": "Превью",
        "refs_sub": "Паттерны → примеры → ссылки",
        "label_why": "Зачем:",
        "label_examples": "Примеры:",
        "label_look": "Смотри в референсах:",
        "label_links": "Ссылки:",
    },
}


def t(uid: int, key: str) -> str:
    return TXT[lang_for(uid)].get(key, key)


# ----------------------------
# Keyboards
# ----------------------------

def main_menu_kb(uid: int) -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton(text=t(uid, "menu_submit"))],
        [KeyboardButton(text=t(uid, "menu_how"))],
    ]
    if is_admin_uid(uid):
        keyboard.append([KeyboardButton(text=t(uid, "menu_stats"))])

    keyboard.append([KeyboardButton(text=t(uid, "menu_channel")), KeyboardButton(text=t(uid, "menu_lang"))])

    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        selective=True,
        input_field_placeholder=t(uid, "need_input"),
    )


def cancel_kb(uid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=t(uid, "btn_cancel"), callback_data=f"cancel:{uid}")]]
    )


def channel_inline_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="@prodooktovy", url="https://t.me/prodooktovy")]]
    )


# ----------------------------
# Progress animation (compact + looping)
# ----------------------------

def retro_bar_frame(step: int, width: int = 16) -> str:
    # more “alive” retro bar: bouncing head + shimmer
    pos = step % (width * 2 - 2)
    if pos >= width:
        pos = (width * 2 - 2) - pos

    palette = ["·", "·", "·", "·", "·"]
    inside = [palette[(i + step) % len(palette)] for i in range(width)]
    inside[pos] = "█"
    if pos - 1 >= 0:
        inside[pos - 1] = "▓"
    if pos - 2 >= 0:
        inside[pos - 2] = "▒"
    if pos + 1 < width:
        inside[pos + 1] = "▓"
    if pos + 2 < width:
        inside[pos + 2] = "▒"

    return (
        "┌" + "─" * width + "┐\n"
        "│" + "".join(inside) + "│\n"
        "└" + "─" * width + "┘"
    )


async def animate_progress_until_done(
    anchor: Message,
    uid: int,
    title: str,
    done_event: asyncio.Event,
    tick: float = 0.12,
) -> Message:
    msg = await anchor.answer(
        f"{escape_html(title)}\n<code>{escape_html(retro_bar_frame(0))}</code>",
        reply_markup=cancel_kb(uid),
    )

    step = 0
    while not done_event.is_set():
        if CANCEL_FLAGS.get(uid):
            break
        step += 1
        try:
            await msg.edit_text(
                f"{escape_html(title)}\n<code>{escape_html(retro_bar_frame(step))}</code>",
                reply_markup=cancel_kb(uid),
            )
        except Exception:
            pass
        await asyncio.sleep(tick)

    # Optional: remove inline keyboard when done (Telegram doesn't let you delete other bot msgs reliably)
    try:
        await msg.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    return msg


# ----------------------------
# Figma: fetch thumbnail via oEmbed (public links)
# ----------------------------

FIGMA_RE = re.compile(r"https?://www\.figma\.com/(file|design)/[A-Za-z0-9]+/[^?\s]+.*", re.I)

def is_figma_link(text: str) -> bool:
    return bool(FIGMA_RE.search(text or ""))


def http_get_bytes(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_figma_thumbnail(figma_url: str) -> Optional[bytes]:
    bust = str(int(time.time() * 1000))
    oembed = "https://www.figma.com/oembed?url=" + urllib.parse.quote(figma_url, safe="")
    oembed += "&cb=" + bust

    try:
        raw = http_get_bytes(oembed, timeout=20.0)
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        thumb = data.get("thumbnail_url")
        if not thumb:
            return None
        sep = "&" if "?" in thumb else "?"
        thumb2 = thumb + f"{sep}cb={bust}"
        return http_get_bytes(thumb2, timeout=25.0)
    except Exception:
        return None


# ----------------------------
# Image: ASCII width control
# ----------------------------

def compute_ascii_width_for_mobile(_: bytes) -> int:
    return 34


def enforce_ascii_lines(lines: List[str], width: int, height: int) -> List[str]:
    fixed: List[str] = []
    for ln in (lines or []):
        ln = str(ln).replace("\t", " ")
        if len(ln) > width:
            ln = ln[:width]
        fixed.append(ln.ljust(width))
        if len(fixed) >= height:
            break
    while len(fixed) < height:
        fixed.append(" " * width)
    return fixed


# ----------------------------
# OpenAI / LLM
# ----------------------------

def img_to_data_uri_png(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


def parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start : end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        return None


def llm_request_with_image(prompt_text: str, data_uri: str) -> str:
    if not client:
        raise RuntimeError("No OpenAI client")

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": data_uri},
            ],
        }],
    )
    return getattr(resp, "output_text", "") or ""


def llm_request_text_only(prompt_text: str) -> str:
    if not client:
        raise RuntimeError("No OpenAI client")

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt_text}],
        }],
    )
    return getattr(resp, "output_text", "") or ""


async def llm_review_image(img_bytes: bytes, uid: int, ascii_w: int) -> Optional[Dict[str, Any]]:
    if not client:
        return {"error": "no_llm"}

    language = lang_for(uid)
    data_uri = img_to_data_uri_png(img_bytes)

    # Note: We removed font-guessing. We ask for more concrete, actionable notes instead.
    prompt = f"""
You are a tough-but-fair senior product designer doing a design review.

Return ONLY valid JSON. No markdown. No extra keys.

Language: "{language}" ("en" or "ru").
Be honest, no profanity.

Tasks:
1) Describe what you see on the screenshot (short, concrete).
2) Give verdict + recommendations (actionable, more concrete):
   - Point to specific UI parts by name (header, primary CTA, form fields, error states, pricing block, etc.)
   - Include quick priority tags like [P0]/[P1]/[P2] inside the verdict text.
   - Avoid guessing fonts or hex colors.
3) Provide a score from 1 to 10 (integer).
4) Provide an ASCII wireframe concept that fits exactly within width={ascii_w} characters per line.
   - Provide as an array of strings (ascii_concept).
   - Each line MUST be <= {ascii_w} chars.

JSON schema:
{{
  "language": "{language}",
  "score_10": 7,
  "what_i_see": "...",
  "verdict": "...",
  "ascii_width": {ascii_w},
  "ascii_height": 18,
  "ascii_concept": ["..."]
}}
""".strip()

    try:
        text = await asyncio.to_thread(llm_request_with_image, prompt, data_uri)
        return parse_json_safe(text)
    except Exception:
        return None


# ----------------------------
# References (meaning-based + Pinterest; simplified scope)
# ----------------------------

def build_refs_prompt(lang: str, what_i_see: str, verdict: str) -> str:
    if _lang_is_ru(lang):
        return f"""
Ты — старший продуктовый дизайнер. Подбери точные референсы "по смыслу" (НЕ по визуалу).
Верни СТРОГО JSON, без markdown.

Контекст:
1) Что видишь: {what_i_see}
2) Вердикт/рекомендации: {verdict}

Выход (строго JSON):
{{
  "items": [
    {{
      "pattern": "название паттерна",
      "why": "почему релевантно (1 строка)",
      "what_to_look_for": ["3–5 деталей"],
      "example_products": ["Product A", "Product B", "Product C"],
      "search_keywords": ["точный запрос (EN)", "ещё запрос (EN)"]
    }}
  ]
}}

Правила:
- 5–7 items
- НЕ пытайся угадать домен/платформу. Просто делай максимально точные ключевые запросы (EN) по паттерну и компонентам.
- search_keywords: английский, конкретно (pattern + components + constraint)
- example_products: реальные продукты/дизайн-системы/гайдлайны
- Никаких ссылок. Только JSON.
""".strip()
    else:
        return f"""
You are a senior product designer. Find precise meaning-based references (not visually similar).
Return STRICT JSON only. No markdown.

Context:
1) What you see: {what_i_see}
2) Verdict / recommendations: {verdict}

Output (strict JSON):
{{
  "items": [
    {{
      "pattern": "pattern name",
      "why": "why it fits (one line)",
      "what_to_look_for": ["3–5 traits"],
      "example_products": ["Product A", "Product B", "Product C"],
      "search_keywords": ["precise EN query", "another EN query"]
    }}
  ]
}}

Rules:
- 5–7 items
- Don't guess domain/platform. Focus on very specific EN queries (pattern + components + constraints).
- No links. JSON only.
""".strip()


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0].strip()
    try:
        return json.loads(s)
    except Exception:
        return parse_json_safe(s)


def _make_link(title: str, url: str) -> str:
    title = escape_html(title or "")
    url = (url or "").replace("&", "&amp;").replace("<", "").replace(">", "")
    return f'• <a href="{url}">{title}</a>'


def build_reference_links(keywords: List[str]) -> List[str]:
    kws = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    picked = kws[:2]
    q = quote_plus(" ".join(picked)) if picked else ""
    if not q:
        q = quote_plus("ux ui pattern")

    # Only 3 searches (as you requested)
    return [
        _make_link("Pinterest", f"https://www.pinterest.com/search/pins/?q={q}"),
        _make_link("Dribbble", f"https://dribbble.com/search/{q}"),
        _make_link("Google (pattern)", f"https://www.google.com/search?q={q}+ui+pattern"),
    ]


def format_refs_block(uid: int, refs: Dict[str, Any]) -> str:
    title = t(uid, "refs")
    subtitle = t(uid, "refs_sub")

    items = (refs or {}).get("items", [])
    if not isinstance(items, list):
        items = []
    items = items[:7]

    lines: List[str] = [f"<b>{escape_html(title)}</b>", f"<i>{escape_html(subtitle)}</i>"]

    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            continue
        pattern = str(it.get("pattern") or "").strip()
        if not pattern:
            continue

        why = str(it.get("why") or "").strip()
        ex = it.get("example_products") or []
        ex = [str(x).strip() for x in ex if str(x).strip()][:4]
        w = it.get("what_to_look_for") or []
        w = [str(x).strip() for x in w if str(x).strip()][:5]
        kws = it.get("search_keywords") or []
        kws = [str(x).strip() for x in kws if str(x).strip()][:4]

        lines.append(f"\n<b>{i}) {escape_html(pattern)}</b>")
        if why:
            lines.append(f"{escape_html(t(uid,'label_why'))} {escape_html(why)}")
        if w:
            lines.append(f"{escape_html(t(uid,'label_look'))} {escape_html('; '.join(w))}")
        if ex:
            lines.append(f"{escape_html(t(uid,'label_examples'))} {escape_html(', '.join(ex))}")

        link_lines = build_reference_links(kws)
        lines.append(f"<u>{escape_html(t(uid,'label_links'))}</u>")
        lines.extend(link_lines)

    return "\n".join(lines).strip()


async def llm_build_references(uid: int, what_i_see: str, verdict: str) -> Optional[Dict[str, Any]]:
    if not client:
        return {"error": "no_llm"}

    prompt = build_refs_prompt(lang_for(uid), what_i_see, verdict)
    try:
        raw = await asyncio.to_thread(llm_request_text_only, prompt)
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            return None
        if "items" not in data or not isinstance(data.get("items"), list) or not data["items"]:
            return None
        return data
    except Exception:
        return None


# ----------------------------
# Review pipeline
# ----------------------------

async def send_channel(uid: int, m: Message) -> None:
    text = f"{t(uid,'channel_msg')} @prodooktovy"
    await m.answer(escape_html(text), reply_markup=channel_inline_kb())


async def do_review(message: Message, bot: Bot, img_bytes: bytes, uid: int, input_type: str) -> None:
    CANCEL_FLAGS[uid] = False
    username = USERNAMES.get(uid, "")
    lang = lang_for(uid)

    t0 = time.time()
    await log_event(user_id=uid, username=username, lang=lang, event="review_started", input_type=input_type)

    done = asyncio.Event()
    spinner_task = asyncio.create_task(
        animate_progress_until_done(
            message,
            uid,
            title=t(uid, "progress_title"),
            done_event=done,
            tick=0.12,
        )
    )

    try:
        if not client:
            done.set()
            await message.answer(escape_html(t(uid, "no_llm")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_failed", input_type=input_type, error="no_openai_key")
            return

        ascii_w = compute_ascii_width_for_mobile(img_bytes)
        result = await llm_review_image(img_bytes, uid, ascii_w)

        done.set()
        try:
            await spinner_task
        except Exception:
            pass

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_cancelled", input_type=input_type)
            return

        if not result:
            await message.answer(escape_html(t(uid, "llm_fail")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_failed", input_type=input_type, error="llm_no_json")
            return

        if result.get("error") == "no_llm":
            await message.answer(escape_html(t(uid, "no_llm")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_failed", input_type=input_type, error="no_llm")
            return

        # enforce language
        result["language"] = lang_for(uid)

        try:
            score_int = int(result.get("score_10", 0))
        except Exception:
            score_int = 0
        score_int = max(1, min(10, score_int))

        what_i_see_text = str(result.get("what_i_see", "")).strip()
        verdict_text = str(result.get("verdict", "")).strip()

        # 1) What I see
        msg1 = (
            f"<b>{escape_html(t(uid,'whatisee'))}</b>\n"
            f"{escape_html(what_i_see_text)}\n\n"
            f"<b>{escape_html(t(uid,'score'))}: {score_int}/10</b>"
        )
        await message.answer(msg1)

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_cancelled", input_type=input_type)
            return

        # 2) Verdict
        msg2 = f"<b>{escape_html(t(uid,'verdict'))}</b>\n{escape_html(verdict_text)}"
        await message.answer(msg2)

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_cancelled", input_type=input_type)
            return

        # 3) ASCII wireframe (spinner)
        done2 = asyncio.Event()
        spinner2 = asyncio.create_task(
            animate_progress_until_done(
                message,
                uid,
                title=t(uid, "progress_wire"),
                done_event=done2,
                tick=0.12,
            )
        )

        try:
            width = int(result.get("ascii_width") or ascii_w)
            height = int(result.get("ascii_height") or 18)
            width = max(26, min(42, width))
            height = max(12, min(26, height))
            lines = result.get("ascii_concept") or []
            if not isinstance(lines, list):
                lines = []
            fixed = enforce_ascii_lines([str(x) for x in lines], width=width, height=height)
            concept_block = "\n".join(fixed)
        finally:
            done2.set()
            try:
                await spinner2
            except Exception:
                pass

        msg3 = f"<b>{escape_html(t(uid,'concept'))}</b>\n<code>{escape_html(concept_block)}</code>"
        await message.answer(msg3)

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_cancelled", input_type=input_type)
            return

        # 4) References (spinner)
        done3 = asyncio.Event()
        spinner3 = asyncio.create_task(
            animate_progress_until_done(
                message,
                uid,
                title=t(uid, "progress_refs"),
                done_event=done3,
                tick=0.12,
            )
        )

        try:
            refs = await llm_build_references(uid, what_i_see_text, verdict_text)
        finally:
            done3.set()
            try:
                await spinner3
            except Exception:
                pass

        if CANCEL_FLAGS.get(uid):
            await message.answer(escape_html(t(uid, "cancelled")), reply_markup=main_menu_kb(uid))
            await log_event(user_id=uid, username=username, lang=lang, event="review_cancelled", input_type=input_type)
            return

        if not refs or refs.get("error") == "no_llm":
            await message.answer(escape_html(t(uid, "refs_fail")), reply_markup=main_menu_kb(uid))
        else:
            refs_msg = format_refs_block(uid, refs)
            await message.answer(refs_msg, disable_web_page_preview=True)

        # end menu
        await message.answer(escape_html(t(uid, "need_input")), reply_markup=main_menu_kb(uid))

        dur_ms = int((time.time() - t0) * 1000)
        await log_event(
            user_id=uid,
            username=username,
            lang=lang,
            event="review_done",
            input_type=input_type,
            duration_ms=dur_ms,
            score=score_int,
        )

    except asyncio.CancelledError:
        # cancelled by user
        await log_event(user_id=uid, username=username, lang=lang, event="review_cancelled", input_type=input_type)
        raise
    except Exception as e:
        try:
            await message.answer(escape_html(t(uid, "llm_fail")), reply_markup=main_menu_kb(uid))
        except Exception:
            pass
        await log_event(user_id=uid, username=username, lang=lang, event="review_failed", input_type=input_type, error=str(e)[:200])
    finally:
        done.set()
        if not spinner_task.done():
            spinner_task.cancel()


# ----------------------------
# Telegram handlers
# ----------------------------

router = Router()


@router.callback_query(F.data.startswith("cancel:"))
async def on_cancel(cb: CallbackQuery):
    try:
        uid = cb.from_user.id
    except Exception:
        return

    CANCEL_FLAGS[uid] = True

    task = RUNNING_TASK.get(uid)
    if task and not task.done():
        task.cancel()

    try:
        await cb.answer("OK", show_alert=False)
    except Exception:
        pass

    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass


@router.message(CommandStart())
async def on_start(m: Message):
    uid = m.from_user.id
    USERNAMES[uid] = (m.from_user.username or "").strip().lower()

    if uid not in LANG:
        set_lang(uid, "en")

    await log_event(user_id=uid, username=USERNAMES[uid], lang=lang_for(uid), event="start")

    await m.answer(
        escape_html(t(uid, "start")),
        reply_markup=main_menu_kb(uid),
    )


@router.message(F.text)
async def on_text(m: Message, bot: Bot):
    uid = m.from_user.id
    USERNAMES[uid] = (m.from_user.username or "").strip().lower()

    txt = (m.text or "").strip()

    # admin-only stats via button
    if txt == t(uid, "menu_stats"):
        if not is_admin_message(m):
            return
        s = await get_stats_summary()
        msg = (
            "<b>Stats</b>\n"
            f"Users: <b>{s['total_users']}</b>\n"
            f"Reviews (total): <b>{s['total_reviews']}</b>\n\n"
            f"Reviews 24h: <b>{s['reviews_24h']}</b>\n"
            f"Reviews 7d: <b>{s['reviews_7d']}</b>\n"
            f"7d by type: photo <b>{s['photo_7d']}</b> / figma <b>{s['figma_7d']}</b>\n"
            f"7d cancels: <b>{s['cancels_7d']}</b>\n"
            f"7d fails: <b>{s['fails_7d']}</b>\n"
        )
        if s["top_errors"]:
            msg += "\n<b>Top errors (7d)</b>\n"
            for e, c in s["top_errors"]:
                msg += f"• {escape_html(str(e))} — {c}\n"
        await m.answer(msg, reply_markup=main_menu_kb(uid))
        return

    # /stats command
    if txt.lower() == "/stats":
        if not is_admin_message(m):
            return
        s = await get_stats_summary()
        msg = (
            "<b>Stats</b>\n"
            f"Users: <b>{s['total_users']}</b>\n"
            f"Reviews (total): <b>{s['total_reviews']}</b>\n\n"
            f"Reviews 24h: <b>{s['reviews_24h']}</b>\n"
            f"Reviews 7d: <b>{s['reviews_7d']}</b>\n"
            f"7d by type: photo <b>{s['photo_7d']}</b> / figma <b>{s['figma_7d']}</b>\n"
            f"7d cancels: <b>{s['cancels_7d']}</b>\n"
            f"7d fails: <b>{s['fails_7d']}</b>\n"
        )
        await m.answer(msg, reply_markup=main_menu_kb(uid))
        return

    # menu actions
    if txt == t(uid, "menu_how"):
        await m.answer(escape_html(t(uid, "how")), reply_markup=main_menu_kb(uid))
        return

    if txt == t(uid, "menu_lang"):
        new_lang = "ru" if lang_for(uid) == "en" else "en"
        set_lang(uid, new_lang)
        await m.answer(escape_html(t(uid, "start")), reply_markup=main_menu_kb(uid))
        return

    if txt == t(uid, "menu_channel"):
        await send_channel(uid, m)
        return

    if txt == t(uid, "menu_submit"):
        await m.answer(escape_html(t(uid, "need_input")))
        return

    # figma link
    if is_figma_link(txt):
        if RUNNING_TASK.get(uid) and not RUNNING_TASK[uid].done():
            await m.answer(escape_html(t(uid, "busy")))
            return

        thumb = fetch_figma_thumbnail(txt)
        if not thumb:
            await m.answer(escape_html(t(uid, "figma_fetch_fail")), reply_markup=main_menu_kb(uid))
            return

        # preview
        try:
            photo = BufferedInputFile(thumb, filename="figma_preview.png")
            await m.answer_photo(photo=photo, caption=escape_html(t(uid, "preview")))
        except Exception:
            pass

        async def _run():
            await do_review(m, bot, thumb, uid, input_type="figma")

        task = asyncio.create_task(_run())
        RUNNING_TASK[uid] = task
        return

    await m.answer(escape_html(t(uid, "not_image")), reply_markup=main_menu_kb(uid))


@router.message(F.photo)
async def on_photo(m: Message, bot: Bot):
    uid = m.from_user.id
    USERNAMES[uid] = (m.from_user.username or "").strip().lower()

    if RUNNING_TASK.get(uid) and not RUNNING_TASK[uid].done():
        await m.answer(escape_html(t(uid, "busy")))
        return

    photo = m.photo[-1]
    file = await bot.get_file(photo.file_id)
    img_bytes = await bot.download_file(file.file_path)
    data = img_bytes.read() if hasattr(img_bytes, "read") else bytes(img_bytes)

    async def _run():
        await do_review(m, bot, data, uid, input_type="photo")

    task = asyncio.create_task(_run())
    RUNNING_TASK[uid] = task


# ----------------------------
# App entry
# ----------------------------

async def main():
    dp = Dispatcher()
    dp.include_router(router)

    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())