
from typing import Dict, Any, List
import json

def load_rules(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def rules_overview(data: Dict[str, Any]) -> str:
    out = ["<b>Правила АвтоДушнилы</b>"]
    chapters = data.get("chapters", [])
    for idx, ch in enumerate(chapters, start=1):
        count = len(ch.get("rules", []))
        out.append(f"{idx}. {ch.get('title','Глава')} — {count} правил")
    out.append("\nОтправьте команду /rules_full чтобы увидеть примеры «было/стало».")
    return "\n".join(out)

def rules_full(data: Dict[str, Any]) -> str:
    out: List[str] = []
    for ch in data.get("chapters", []):
        out.append(f"📘 <b>{ch.get('title','Глава')}</b>")
        for r in ch.get("rules", []):
            ex = r.get("examples", {})
            bad = ex.get("bad"); good = ex.get("good")
            ex_block = ""
            if bad or good:
                ex_block = f"\nБыло: {bad or '—'}\nСтало: {good or '—'}"
            out.append(f"{r.get('id','?')} — {r.get('name','Правило')}\n{r.get('description','')}{ex_block}")
        out.append("")
    txt = "\n".join(out).strip()
    return txt
