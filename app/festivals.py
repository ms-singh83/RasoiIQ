"""Festival and holiday calendar for Chandigarh / Punjab, 2025-2026.

Lunar festivals move every year, so treat these as best-effort and edit freely.
A CSV row whose notes mention a festival, party or function also counts.
"""
from __future__ import annotations

from datetime import date

FESTIVALS: dict[date, str] = {
    date(2025, 1, 13): "Lohri",
    date(2025, 1, 26): "Republic Day",
    date(2025, 3, 14): "Holi",
    date(2025, 3, 31): "Eid ul-Fitr",
    date(2025, 4, 13): "Baisakhi",
    date(2025, 8, 9): "Raksha Bandhan",
    date(2025, 8, 15): "Independence Day",
    date(2025, 8, 16): "Janmashtami",
    date(2025, 9, 22): "Navratri begins",
    date(2025, 10, 2): "Dussehra",
    date(2025, 10, 10): "Karva Chauth",
    date(2025, 10, 20): "Diwali",
    date(2025, 10, 22): "Bhai Dooj",
    date(2025, 11, 5): "Guru Nanak Jayanti",
    date(2025, 12, 25): "Christmas",
    date(2025, 12, 31): "New Year's Eve",
    date(2026, 1, 13): "Lohri",
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 4): "Holi",
    date(2026, 3, 20): "Eid ul-Fitr",
    date(2026, 4, 14): "Baisakhi",
    date(2026, 5, 27): "Eid al-Adha",
    date(2026, 8, 15): "Independence Day",
    date(2026, 8, 28): "Raksha Bandhan",
    date(2026, 9, 4): "Janmashtami",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 2): "Gandhi Jayanti",
    date(2026, 10, 11): "Navratri begins",
    date(2026, 10, 20): "Dussehra",
    date(2026, 10, 29): "Karva Chauth",
    date(2026, 11, 8): "Diwali",
    date(2026, 11, 10): "Bhai Dooj",
    date(2026, 11, 24): "Guru Nanak Jayanti",
    date(2026, 12, 25): "Christmas",
    date(2026, 12, 31): "New Year's Eve",
}

NOTE_KEYWORDS = ("festival", "tyohar", "party", "holiday", "puja", "function", "wedding", "shaadi")


def festival_name(d: date) -> str | None:
    return FESTIVALS.get(d)


def note_marks_festival(note: object) -> bool:
    return isinstance(note, str) and any(k in note.lower() for k in NOTE_KEYWORDS)
