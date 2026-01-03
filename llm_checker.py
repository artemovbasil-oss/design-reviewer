
from typing import List, Dict, Any, Optional
import json, os

def load_rules(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat = {}
    for ch in data.get("chapters", []):
        for r in ch.get("rules", []):
            flat[r["id"]] = {"chapter": ch["title"], **r}
    data["_index"] = flat
    return data

def _match_patterns(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    for p in patterns or []:
        if p and p.lower() in t:
            return True
    return False

def check_lines_against_rules(lines: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    for i, line_obj in enumerate(lines):
        line_text = (line_obj.get("text") or "").strip()
        if not line_text:
            continue
        for ch in rules.get("chapters", []):
            title = ch.get("title","")
            if title.startswith("5."):
                continue
            for r in ch.get("rules", []):
                pats = r.get("patterns", [])
                matched = False
                if pats:
                    matched = _match_patterns(line_text, pats)
                else:
                    if r.get("id") == "1.2":
                        matched = len(line_text.split()) > 18
                    elif r.get("id") == "1.6":
                        low = line_text.lower()
                        matched = ("подтверд" in low and "подтвержден" in low) or ("оплат" in low and "оплата" in low)
                    else:
                        matched = False
                if matched:
                    findings.append({
                        "rule_id": r["id"],
                        "rule_name": r.get("name",""),
                        "chapter": title,
                        "line": line_text,
                        "line_index": i,
                        "bbox": line_obj.get("bbox"),
                        "evidence": r.get("description",""),
                        "examples": r.get("examples", {}),
                    })
    findings.sort(key=lambda x: (x["line_index"], x["rule_id"]))
    return findings

def format_findings(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "Нарушений не найдено."
    out = []
    for f in findings:
        rid = f.get("rule_id","?")
        rname = f.get("rule_name","Правило")
        chap = f.get("chapter","Глава")
        frag = f.get("line","")
        ex = f.get("examples") or {}
        ex_bad = ex.get("bad"); ex_good = ex.get("good")
        ex_block = ""
        if ex_bad or ex_good:
            ex_block = "\nПример:\n— Было: " + (ex_bad or "—") + "\n— Стало: " + (ex_good or "—")
        out.append(f"• Нарушено правило {rid} — {rname}\nГлава: {chap}\nФрагмент: «{frag}»{ex_block}")
    return "\n\n".join(out)

def _init_openai_client():
    try:
        import openai  # type: ignore
    except Exception:
        return None, False
    if not getattr(openai, "api_key", None):
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_TOKEN")
        if key:
            try: openai.api_key = key
            except Exception: pass
    return openai, bool(getattr(openai, "api_key", None))

def _chat(openai, model: str, messages: List[Dict[str,str]], temperature: float, max_tokens: int) -> str:
    try:
        resp = openai.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content.strip()
    except Exception:
        pass
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def llm_brief_description(lines: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> str:
    openai, ok = _init_openai_client()
    if not ok:
        return ""
    text = "\n".join([(l.get("text") or "") for l in lines])
    return _chat(openai, model, [
        {"role":"system","content":"Опиши кратко экран (2–3 предложения), только факты без домыслов."},
        {"role":"user","content": text[:4000]}
    ], temperature=0.2, max_tokens=250)

def llm_plain_recommendations(text: str, findings: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> str:
    openai, ok = _init_openai_client()
    if not ok:
        return ""
    refs = "\n".join([f"{f.get('rule_id','?')} — {f.get('rule_name','')}" for f in findings[:20]])
    return _chat(openai, model, [
        {"role":"system","content":"Ты UX‑редактор. Пиши кратко и по делу, русский язык."},
        {"role":"user","content": f"Текст:\n{text[:5000]}\n\nНарушения:\n{refs}\n\nДай правки с ссылками на правила."}
    ], temperature=0.3, max_tokens=700)

def llm_ux_audit(text: str, lines: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> str:
    openai, ok = _init_openai_client()
    if not ok:
        return ""
    joined = text or "\n".join([(l.get("text") or "") for l in lines])
    return _chat(openai, model, [
        {"role":"system","content":"Ты опытный UX‑консультант по финтеху. Дай 5–10 пунктов UX‑аудита, кратко."},
        {"role":"user","content": "Текст на экране:\n\n" + joined[:6000]}
    ], temperature=0.3, max_tokens=700)

def llm_generate_screen(text: str, ux_feedback: str, model: str = "gpt-4o-mini") -> str:
    openai, ok = _init_openai_client()
    if not ok:
        return ""
    prompt = (
        "Ты продуктовый дизайнер. Сгенерируй текстовую спецификацию улучшённого экрана по формату:"
        "\n- Заголовок\n- Подзаголовок\n- Основная кнопка\n- Вторичная кнопка (если нужна)"
        "\n- Поля (Название/Плейсхолдер/Хинт/Проверка)\n- Микротекст\n- Подвал"
    )
    user = (
        "Исходный текст:\n" + (text or "") +
        "\n\nUX‑замечания:\n" + (ux_feedback or "") +
        "\n\nСоблюдай краткость и структурность."
    )
    return _chat(openai, model, [
        {"role":"system","content":"Русский язык, лаконично, без воды."},
        {"role":"user","content": prompt + "\n\n" + user}
    ], temperature=0.3, max_tokens=1000)
