"""Prep plan and shopping list from the forecast.

All numbers are computed here in Python. If GEMMA_API_KEY is set, Gemma only
rewrites the finished list in friendly Hinglish, and it is sent just the
plan (dish names and counts), never the order history.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass

import httpx

from .config import FRIEND_NAME
from .forecast import ForecastResult

log = logging.getLogger("rasoiiq.planner")

# Rough grams per plate. Dishes not listed here still get a prep count, just no shopping lines.
RECIPES: dict[str, dict[str, float]] = {
    "Dal Makhani": {"Urad dal (sabut)": 60, "Rajma": 10, "Butter": 15, "Cream": 20, "Tomato": 60},
    "Rajma Chawal": {"Rajma": 70, "Basmati rice": 90, "Onion": 50, "Tomato": 60},
    "Aloo Paratha": {"Atta": 120, "Aloo": 150, "Butter": 10, "Dahi": 50},
    "Paneer Butter Masala": {"Paneer": 120, "Tomato": 100, "Butter": 15, "Cream": 25, "Kaju": 10},
    "Gajar Halwa": {"Gajar": 250, "Doodh": 200, "Cheeni": 40, "Ghee": 15, "Khoya": 30},
}


@dataclass
class Plan:
    text: str
    source: str  # "template" or "gemma:<model>"
    note: str | None
    shopping: list[dict]


def shopping_list(result: ForecastResult) -> list[dict]:
    totals: dict[str, float] = {}
    for it in result.items:
        for ing, grams in RECIPES.get(it.item, {}).items():
            totals[ing] = totals.get(ing, 0.0) + grams * it.prep
    out = []
    for ing, g in sorted(totals.items(), key=lambda kv: -kv[1]):
        g = math.ceil(g / 50) * 50  # round up to the nearest 50 g
        if g >= 1000:
            out.append({"ingredient": ing, "qty": math.ceil(g / 100) / 10, "unit": "kg"})
        else:
            out.append({"ingredient": ing, "qty": int(g), "unit": "g"})
    return out


def template_plan(result: ForecastResult, shopping: list[dict], festival: str | None) -> str:
    day = result.target_date.strftime("%A, %d %b")
    lines = [f"Prep plan for {day}:"]
    for it in result.items:
        rng = f" (likely {math.floor(it.p10)}-{math.ceil(it.p90)})" if it.p10 is not None else ""
        lines.append(f"- {it.item}: {it.prep} plates{rng}")
    lines.append("Shopping list:")
    if shopping:
        lines += [f"- {s['ingredient']}: {s['qty']} {s['unit']}" for s in shopping]
    else:
        lines.append("- Add your dishes to RECIPES in app/planner.py to get quantities here.")
    if festival:
        lines.append(f"Note: tomorrow is marked as {festival}, so the numbers already include the festival bump.")
    return "\n".join(lines)


def gemma_prompt(plan_text: str) -> str:
    return (
        f"You help {FRIEND_NAME}, who runs a small home kitchen in India. Rewrite this prep plan "
        "and shopping list in simple, warm Hinglish (Hindi in Roman script mixed with English). "
        "Keep every dish, ingredient and number exactly the same. Do not add new numbers. "
        "Keep it as a short list, no markdown, no emojis. End with one practical tip in one line.\n\n"
        f"{plan_text}"
    )


def numbers_preserved(original: str, rewritten: str) -> bool:
    """Every number in our plan must appear in Gemma's version."""
    want = re.findall(r"\d+(?:\.\d+)?", original)
    have = set(re.findall(r"\d+(?:\.\d+)?", rewritten))
    return all(n in have for n in want)


def gemma_rewrite(text: str, api_key: str) -> tuple[str, str]:
    model = os.getenv("GEMMA_MODEL", "gemma-3-27b-it")
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key},
        json={"contents": [{"role": "user", "parts": [{"text": gemma_prompt(text)}]}],
              "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}},
        timeout=float(os.getenv("GEMMA_TIMEOUT", "45")),
    )
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    out = "".join(p.get("text", "") for p in parts).replace("**", "").strip()
    return out, f"gemma:{model}"


def make_plan(result: ForecastResult, festival: str | None = None) -> Plan:
    shopping = shopping_list(result)
    text = template_plan(result, shopping, festival)
    key = os.getenv("GEMMA_API_KEY")
    if not key:
        return Plan(text, "template", None, shopping)
    try:
        rewritten, source = gemma_rewrite(text, key)
    except Exception as exc:
        log.warning("Gemma rewrite failed: %s", exc)
        return Plan(text, "template", f"Gemma was unreachable ({type(exc).__name__}), showing the plain plan.", shopping)
    if not rewritten or not numbers_preserved(text, rewritten):
        return Plan(text, "template", "Gemma changed some numbers, so the plain plan is shown instead.", shopping)
    return Plan(rewritten, source, None, shopping)
