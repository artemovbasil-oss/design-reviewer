# Design Reviewer Bot 🤍

A Telegram bot for honest UI/UX design reviews.

This bot behaves like a senior design partner:  
it carefully looks at your interface, calls out weak decisions without sugarcoating, and explains what works well — and why.

The bot accepts **UI screenshots** and **links to Figma frames** (if the file is public), analyzes both visual design and interface copy, and returns a structured critique.

---

## What the bot can review

### 📸 Input formats
- UI screenshots (mobile, web, dashboards)
- Links to Figma frames  
  (`https://www.figma.com/...node-id=...`, the file must be public)

### 🔍 What it analyzes

**1. What is shown on the screen**  
A short, clear description of the interface, context, and user scenario.

**2. Visual design**
- Visual hierarchy and readability
- Layout logic and composition
- Typography impression (font families are *estimated*, no technical measurements)
- Overall clarity, rhythm, and balance
- If something works well — the bot explains *why*
- If something is weak — the bot clearly points it out and suggests improvements

**3. Interface text**
- Clarity and specificity
- Tone and confidence
- Confusing or vague wording
- Concrete suggestions on how to rewrite texts

### 📊 Final score
- Overall UI quality score on a **10-point scale**
- No marketing fluff — only design reasoning

---

## How it works

1. Tap **“Submit for review”** or simply send:
   - a screenshot  
   - or a Figma frame link
2. The bot shows a retro-style ASCII loading animation
3. You receive **three messages**:
   1. What the bot sees on the screen  
   2. Visual design review  
   3. Text review + final score  

---

## Bot interface

Main menu:
- **🖼 Submit for review** — send a screenshot or Figma link
- **❓ How does it work?** — short usage guide
- **◻ Ping** — check that the bot is alive

Only **black-and-white emojis** are used to keep the interface clean.

---

## Tone & philosophy

- The bot is not “nice by default”
- No profanity, but no softening either
- Speaks like a senior colleague, not customer support
- Good decisions are praised and explained
- Bad decisions are called out directly, with suggestions

This is not an automated checklist.  
This is **design review**.

---

## Tech stack (brief)

- Python
- aiogram 3
- OpenAI Vision / LLM
- OCR for text extraction
- Railway for deployment

(Focused on results, not on technical verbosity.)

---

## Author

Created and maintained by  
**Basil Artemov** — Senior Product / UX Designer  

Portfolio: **https://ux.luxury**

---

## License

MIT — feel free to use, fork, and adapt.
