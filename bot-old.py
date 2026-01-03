import os, io, json, re, statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from PIL import Image, ImageDraw
import pytesseract

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    BufferedInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from aiogram.filters import CommandStart, Command
from aiogram.enums.parse_mode import ParseMode
from aiogram.client.default import DefaultBotProperties

from html import escape as htmlesc
from dotenv import load_dotenv

# --- Надёжная загрузка .env (рядом с bot.py) ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
OCR_LANG = os.getenv("OCR_LANG", "rus+eng")
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL")
RULES_PATH = os.getenv("RULES_PATH", "rules.json")

# Пороговые настройки OCR (можно править в .env)
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "55"))   # минимальная средняя уверенность линии
OCR_MIN_WORD_CONF = float(os.getenv("OCR_MIN_WORD_CONF", "45"))  # минимальная уверенность слова
OCR_MIN_LEN = int(os.getenv("OCR_MIN_LEN", "2"))        # минимальная длина строки после нормализации
OCR_MIN_ALPHA_FRAC = float(os.getenv("OCR_MIN_ALPHA_FRAC", "0.45"))  # доля букв/цифр в строке
BUTTON_MAX_WORDS = int(os.getenv("BUTTON_MAX_WORDS", "4"))           # сколько слов максимум для кнопки
HEADING_HEIGHT_MULT = float(os.getenv("HEADING_HEIGHT_MULT", "1.35"))# порог высоты для заголовка

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN in .env or environment")

# --- aiogram setup ---
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# --- Дружелюбная клавиатура (главное меню) ---
main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🔍 Проверить текст"), KeyboardButton(text="🖼 Проверить скрин")],
        [KeyboardButton(text="📘 Правила"), KeyboardButton(text="⚙️ Настройки")],
    ],
    resize_keyboard=True,
    input_field_placeholder="Выбери действие 👇",
)

# --- LLM мягкие проверки (опционально) ---
import llm_checker

# --- Загрузка правил ---
with open(RULES_PATH, "r", encoding="utf-8") as f:
    RULES_DB: Dict[str, Any] = json.load(f)

# ===================== НОРМАЛИЗАЦИЯ ТЕКСТА =====================

LATIN_TO_CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у"
})

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.translate(LATIN_TO_CYR)
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("“", "\"").replace("”", "\"").replace("«", "\"").replace("»", "\"")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def frac_alnum(s: str) -> float:
    if not s:
        return 0.0
    al = sum(ch.isalnum() for ch in s)
    return al / max(1, len(s))

# ===================== ДОП. РЕГУЛЯРКИ-«ЛОВУШКИ» =====================

EXTRA_FORBIDDEN_REGEX = [
    r"\bвнимани[её]\b\s*!*",
    r"\bпроизошла?\s+ошибк[аи]\b\s*!*",
    r"\bошибк[аи]\b\s*!*",
    r"\bуважаем\w*\b",
    r"\bк\s*сожален\w*\b",
    r"\bуспешн\w*\b",
    r"\bотправлен[оа]?\s+в\s+обработк\w*\b",
    r"\bожидается\s+подтвержден\w*\b",
    r"\bне\s+исполнен\w*\b",
    r"\bошибк[аи]\s*\d{3,}\b",
    r"\binvalid\s+token\b",
]

# ===================== OCR (координаты в масштабе оригинала) =====================

def ocr_with_boxes(pil_img: Image.Image, lang: str = OCR_LANG):
    orig_w, orig_h = pil_img.size
    scale = 1.5 if max(orig_w, orig_h) < 1600 else 1.0
    work_img = pil_img if scale == 1.0 else pil_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)

    gray = work_img.convert("L")
    bw = gray.point(lambda x: 255 if x > 180 else 0, mode="1").convert("L")

    custom_config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(bw, lang=lang, config=custom_config, output_type=pytesseract.Output.DICT)

    inv = 1.0 / scale
    words = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        # фильтр по уверенности слова
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < OCR_MIN_WORD_CONF:
            continue

        x_s, y_s = int(data["left"][i]), int(data["top"][i])
        w_s, h_s = int(data["width"][i]), int(data["height"][i])

        x = int(x_s * inv); y = int(y_s * inv)
        w = max(1, int(w_s * inv)); h = max(1, int(h_s * inv))

        x = max(0, min(x, orig_w - 1)); y = max(0, min(y, orig_h - 1))
        if x + w > orig_w: w = orig_w - x
        if y + h > orig_h: h = orig_h - y

        words.append({"text": txt, "bbox": (x, y, w, h), "conf": conf})
    return words

def merge_words_to_lines(words, y_thresh=12):
    # предварительно отсечём явный мусор по символам
    clean = []
    for w in words:
        txt = normalize_text(w["text"])
        if len(txt) < 1:
            continue
        if frac_alnum(txt) < 0.3 and len(txt) < 3:
            continue
        clean.append({**w, "text": txt})

    words_sorted = sorted(clean, key=lambda w: (w["bbox"][1], w["bbox"][0]))
    lines, current, last_y = [], [], None
    for w in words_sorted:
        y = w["bbox"][1]
        if last_y is None or abs(y - last_y) <= y_thresh:
            current.append(w)
            if last_y is None:
                last_y = y
        else:
            lines.append(current)
            current = [w]
            last_y = y
    if current:
        lines.append(current)

    result = []
    for line in lines:
        text = " ".join([w["text"] for w in line]).strip()
        if not text:
            continue
        xs = [w["bbox"][0] for w in line]
        ys = [w["bbox"][1] for w in line]
        ws = [w["bbox"][2] for w in line]
        hs = [w["bbox"][3] for w in line]
        confs = [w["conf"] for w in line]

        x0, y0 = min(xs), min(ys)
        x1 = max(xs[i] + ws[i] for i in range(len(xs)))
        y1 = max(ys[i] + hs[i] for i in range(len(ys)))
        avg_conf = sum(confs) / max(1, len(confs))

        # финальная фильтрация линии
        if avg_conf < OCR_MIN_CONF:
            continue
        if len(text) < OCR_MIN_LEN:
            continue
        if frac_alnum(text.lower()) < OCR_MIN_ALPHA_FRAC:
            continue

        result.append({"text": text, "bbox": (x0, y0, x1 - x0, y1 - y0), "avg_conf": avg_conf, "height": (y1 - y0)})
    return result

# ===================== КЛАССИФИКАЦИЯ СТРОК =====================

ACTION_VERBS = {
    "подтвердить","отправить","удалить","оплатить","создать","продолжить","изменить","начать",
    "вернуться","получить","повторить","исправить","скачать","открыть","пополнить","проверить",
    "перейти","добавить","подписать","сохранить","выбрать","завершить","оформить","активировать","подключить"
}

def classify_lines(lines: List[Dict[str, Any]]) -> None:
    """Добавляет поле role: heading|button|text, чтобы корректно применять правила."""
    if not lines:
        return
    heights = [ln["height"] for ln in lines]
    median_h = statistics.median(heights)

    for ln in lines:
        t = ln["text"]
        tl = t.lower()
        wc = len(t.split())
        h_ratio = (ln["height"] / max(1.0, median_h))

        looks_heading = (h_ratio >= HEADING_HEIGHT_MULT and wc >= 2) or (wc >= 4 and h_ratio >= 1.15)
        # кнопка: короткий текст + начинается с глагола-действия ИЛИ вся строка = 1-3 слова без точки
        starts_with_action = any(tl.startswith(v) for v in ACTION_VERBS)
        looks_button = (wc <= BUTTON_MAX_WORDS and not t.endswith(".") and (starts_with_action or wc <= 3))

        if looks_button and not looks_heading:
            ln["role"] = "button"
        elif looks_heading:
            ln["role"] = "heading"
        else:
            ln["role"] = "text"

# ===================== ПРАВИЛА =====================

@dataclass
class Violation:
    rule_id: str
    title: str
    severity: str
    description: str
    suggestion: str
    text: str
    bbox: Tuple[int, int, int, int]
    kind: str  # "hard" | "soft"

def _match_any_regex(text: str, regex_list: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in (regex_list or []))

def _match_any_exact(text: str, items: List[str]) -> bool:
    t = text.lower()
    return any((it or "").strip().lower() in t for it in (items or []))

def apply_rules(lines: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Violation]:
    violations: List[Violation] = []
    for line in lines:
        t_raw = line["text"]           # уже нормализованная строка
        t = t_raw.lower()
        bbox = line["bbox"]
        role = line.get("role", "text")

        for r in rules.get("rules", []):
            rid, title, severity = r.get("id", ""), r.get("title", ""), r.get("severity", "low")
            desc, sugg = r.get("description", ""), r.get("suggestion", "")
            applies_to = r.get("applies_to")  # может быть ["button"]

            # Если правило ограничено типом элемента — уважаем это
            if applies_to:
                if "button" in applies_to and role != "button":
                    continue

            # Жёсткие запреты
            hard_hit = False
            if _match_any_exact(t, r.get("patterns_forbidden")):
                hard_hit = True
            if _match_any_regex(t, r.get("patterns_forbidden_regex")):
                hard_hit = True
            if not hard_hit and any(re.search(rx, t, flags=re.IGNORECASE) for rx in EXTRA_FORBIDDEN_REGEX):
                # доп. ловушки применяем только к обычному тексту/статусам/заголовкам
                if role != "button":  # чтобы не перетриггерить кнопки случайно
                    hard_hit = True

            if hard_hit:
                violations.append(Violation(rid, title, severity, desc, sugg, t_raw, bbox, "hard"))
                continue

            # Обязательные шаблоны (только если правило их требует)
            if r.get("patterns_required_any"):
                ok = any(k.strip().lower() in t for k in r["patterns_required_any"])
                if not ok:
                    violations.append(Violation(rid, title, severity, desc, sugg, t_raw, bbox, "hard"))
                    continue

            # Мягкие подсказки
            if r.get("soft_check") and _match_any_regex(t, r.get("patterns_forbidden_regex", [])):
                violations.append(Violation(rid, title, severity, desc, sugg, t_raw, bbox, "soft"))
    return violations

# ===================== ВИЗУАЛИЗАЦИЯ =====================

def draw_annotations(pil_img: Image.Image, violations: List[Violation]) -> Image.Image:
    img = pil_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    for v in violations:
        x, y, w, h = v.bbox
        color = (255, 0, 0, 255) if v.kind == "hard" else (255, 165, 0, 255)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = f"{v.rule_id}"
        tw = draw.textlength(label)
        pad = 4
        top = max(0, y - 18)
        draw.rectangle([x, top, x + int(tw) + 2*pad, top + 16], fill=color)
        draw.text((x + pad, top + 1), label, fill=(255, 255, 255, 255))
    return img

# ===================== ХЕНДЛЕРЫ =====================

@dp.message(CommandStart())
async def start(m: Message):
    await m.answer(
        "👋 Привет! Это Bereke UI Text Checker.\n"
        "• Проверь тексты на соответствие редполитике Bereke\n"
        "• Отправь скрин для OCR-проверки или используй /check\n"
        "• Для отладки OCR — /debug_ocr",
        reply_markup=main_kb,
    )

@dp.message(F.text == "🔍 Проверить текст")
async def shortcut_check(m: Message):
    await m.answer("Отправь текст, который хочешь проверить:")

@dp.message(F.text == "🖼 Проверить скрин")
async def shortcut_image(m: Message):
    await m.answer("Пришли скриншот интерфейса — найду проблемные тексты 🕵️")

@dp.message(F.text == "📘 Правила")
async def shortcut_rules(m: Message):
    await cmd_rules(m)

@dp.message(F.text == "⚙️ Настройки")
async def shortcut_settings(m: Message):
    await m.answer(
        "Настройки OCR:\n"
        f"• OCR_MIN_CONF: <code>{OCR_MIN_CONF}</code>\n"
        f"• OCR_MIN_WORD_CONF: <code>{OCR_MIN_WORD_CONF}</code>\n"
        f"• OCR_MIN_LEN: <code>{OCR_MIN_LEN}</code>\n"
        f"• OCR_MIN_ALPHA_FRAC: <code>{OCR_MIN_ALPHA_FRAC}</code>\n"
        f"• BUTTON_MAX_WORDS: <code>{BUTTON_MAX_WORDS}</code>\n"
        f"• HEADING_HEIGHT_MULT: <code>{HEADING_HEIGHT_MULT}</code>\n"
        "\nИзмени переменные в .env и перезапусти бота.",
    )

@dp.message(Command("rules"))
async def cmd_rules(m: Message):
    meta = RULES_DB.get("meta", {})
    rules = RULES_DB.get("rules", [])
    head = f"<b>Правила:</b> {len(rules)} шт. Источник: {htmlesc(meta.get('source', '?'))}\n"
    lines = []
    for r in rules[:30]:
        rid = htmlesc(r.get("id", ""))
        sev = htmlesc(r.get("severity", "low"))
        title = htmlesc(r.get("title", ""))
        lines.append(f"• <b>{rid}</b> ({sev}): {title}")
    tail = f"\n… и ещё {len(rules) - 30}" if len(rules) > 30 else ""
    await m.answer(head + "\n".join(lines) + tail)

@dp.message(Command("check"))
async def cmd_check(m: Message):
    payload = (m.text or "").partition(" ")[2].strip()
    if not payload:
        await m.answer("Формат: <code>/check &lt;текст&gt;</code>")
        return

    # одна строка — как отдельная "линия"
    lines = [{"text": normalize_text(payload), "bbox": (0, 0, 100, 20), "height": 16}]
    classify_lines(lines)

    llm_issues_map = {}
    if LLM_ENABLED:
        try:
            llm_issues_map = llm_checker.llm_soft_checks([payload], model=LLM_MODEL)
        except Exception:
            llm_issues_map = {}

    violations = apply_rules(lines, RULES_DB)
    if LLM_ENABLED and llm_issues_map.get(0):
        for issue in llm_issues_map[0]:
            violations.append(
                Violation(
                    rule_id=str(issue.get("rule_id", "LLM")),
                    title="LLM-мягкая проверка",
                    severity=str(issue.get("severity", "low")).lower(),
                    description=str(issue.get("note", "")),
                    suggestion=str(issue.get("note", "")),
                    text=payload,
                    bbox=(0, 0, 100, 20),
                    kind="soft",
                )
            )

    if not violations:
        await m.answer("Нарушений не найдено ✅")
        return

    chunks = ["<b>Найдены замечания:</b>"]
    for v in violations[:50]:
        prefix = "🔴" if v.kind == "hard" else "🟠"
        chunks.append(
            f"{prefix} <b>{htmlesc(v.rule_id)}</b> ({htmlesc(v.severity)}): "
            f"«{htmlesc(v.text)}» — {htmlesc(v.title)}. <i>{htmlesc(v.suggestion)}</i>"
        )
    await m.answer("\n".join(chunks))

@dp.message(Command("debug_ocr"))
async def cmd_debug_ocr(m: Message):
    await m.answer("Ок. Теперь ответь на это сообщение КАРТИНКОЙ — пришлю распознанные строки, их роль (heading/button/text) и нормализацию.")

@dp.message(F.photo | F.document[(F.document.mime_type.startswith("image/"))])
async def handle_image(m: Message):
    # Получаем файл
    if m.photo:
        file_id = m.photo[-1].file_id
    else:
        file_id = m.document.file_id
    file = await bot.get_file(file_id)
    f = await bot.download_file(file.file_path)
    img = Image.open(io.BytesIO(f.read())).convert("RGB")

    # OCR → слова → строки
    words = ocr_with_boxes(img, OCR_LANG)
    lines = merge_words_to_lines(words)
    classify_lines(lines)

    # debug-режим
    is_debug_reply = bool(m.reply_to_message and "/debug_ocr" in (m.reply_to_message.text or ""))
    if is_debug_reply:
        debug_lines = []
        for ln in lines[:60]:
            role = ln.get("role", "text")
            debug_lines.append(
                f"• ({role}) «{htmlesc(ln['text'])}»\n   norm: «{htmlesc(normalize_text(ln['text']))}», h={ln.get('height')}, conf≈{int(ln.get('avg_conf',0))}"
            )
        if not debug_lines:
            await m.answer("Ничего не распознано 🤷‍♂️")
        else:
            chunk = []
            for i, row in enumerate(debug_lines, 1):
                chunk.append(row)
                if i % 25 == 0:
                    await m.answer("\n".join(chunk))
                    chunk = []
            if chunk:
                await m.answer("\n".join(chunk))
        return

    # LLM soft checks (опционально)
    llm_issues_map = {}
    if LLM_ENABLED:
        try:
            llm_issues_map = llm_checker.llm_soft_checks([ln["text"] for ln in lines], model=LLM_MODEL)
        except Exception:
            llm_issues_map = {}

    # Правила
    violations = apply_rules(lines, RULES_DB)

    # LLM-подсказки
    if LLM_ENABLED and llm_issues_map:
        for idx, items in llm_issues_map.items():
            if 0 <= idx < len(lines):
                line = lines[idx]
                for issue in items:
                    rid = str(issue.get("rule_id", "LLM"))
                    note = str(issue.get("note", ""))
                    sev = str(issue.get("severity", "low")).lower()
                    violations.append(
                        Violation(
                            rule_id=rid,
                            title="LLM-мягкая проверка",
                            severity=sev,
                            description=note or "",
                            suggestion=note or "Проверьте формулировку по редполитике.",
                            text=line["text"],
                            bbox=line["bbox"],
                            kind="soft",
                        )
                    )

    # Сортировка
    violations_sorted = sorted(
        violations,
        key=lambda v: (0 if v.kind == "hard" else 1, {"high": 0, "medium": 1, "low": 2}.get(v.severity, 2)),
    )

    # Аннотации
    annotated = draw_annotations(img, violations_sorted)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)

    # Отчёт
    if not violations_sorted:
        await m.answer("Нарушений не найдено ✅")
    else:
        chunks = ["<b>Найдены замечания:</b>"]
        for v in violations_sorted[:60]:
            prefix = "🔴" if v.kind == "hard" else "🟠"
            chunks.append(
                f"{prefix} <b>{htmlesc(v.rule_id)}</b> ({htmlesc(v.severity)}): "
                f"«{htmlesc(v.text)}» — {htmlesc(v.title)}. "
                f"<i>{htmlesc(v.suggestion)}</i>"
            )
        text_out = "\n".join(chunks)
        while len(text_out) > 3500:
            cut = text_out[:3500]
            last_nl = cut.rfind("\n")
            if last_nl == -1:
                last_nl = 3500
            await m.answer(cut[:last_nl])
            text_out = text_out[last_nl+1:]
        if text_out:
            await m.answer(text_out)

    await m.answer_photo(BufferedInputFile(buf.getvalue(), filename="annotated.png"))

# ===================== ЗАПУСК =====================

if __name__ == "__main__":
    import asyncio
    print(f"✅ Bereke bot starting... OCR_LANG={OCR_LANG}, LLM_ENABLED={LLM_ENABLED}, model={LLM_MODEL}, rules={RULES_PATH}")
    asyncio.run(dp.start_polling(bot))
