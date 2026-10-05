from datetime import date

import httpx

from app import planner
from app.forecast import ForecastResult, ItemForecast


def _result():
    items = [ItemForecast("Dal Makhani", 10, 8.0, 10.0, 14.0, 12, 11.0),
             ItemForecast("Mystery Dish", 3, 1.0, 3.0, 5.0, 4, None)]
    return ForecastResult(date(2026, 10, 4), items, "fake", None, None, {})


def test_shopping_list_math():
    shop = {s["ingredient"]: (s["qty"], s["unit"]) for s in planner.shopping_list(_result())}
    assert shop["Urad dal (sabut)"] == (750, "g")   # 60 g x 12 plates = 720 -> 750
    assert shop["Tomato"] == (750, "g")
    assert len(shop) == 5                           # unknown dish adds nothing


def test_kg_rounding():
    r = _result()
    r.items[0].prep = 17  # 60 x 17 = 1020 g urad dal
    shop = {s["ingredient"]: (s["qty"], s["unit"]) for s in planner.shopping_list(r)}
    assert shop["Urad dal (sabut)"] == (1.1, "kg")


def test_template_without_key(monkeypatch):
    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    p = planner.make_plan(_result(), festival="Diwali")
    assert p.source == "template" and p.note is None
    assert "Dal Makhani: 12 plates (likely 8-14)" in p.text and "Diwali" in p.text


def test_numbers_preserved():
    assert planner.numbers_preserved("Dal: 12 plates, 750 g", "Dal 12 plate banao, 750 g lao")
    assert not planner.numbers_preserved("Dal: 12 plates", "Dal 13 plates")


def test_gemma_used_when_numbers_match(monkeypatch):
    monkeypatch.setenv("GEMMA_API_KEY", "test")
    monkeypatch.setattr(planner, "gemma_rewrite", lambda text, key: (text.replace("Prep plan", "Kal ka plan"), "gemma:test"))
    p = planner.make_plan(_result())
    assert p.source == "gemma:test" and p.text.startswith("Kal ka plan")


def test_gemma_rejected_when_numbers_change(monkeypatch):
    monkeypatch.setenv("GEMMA_API_KEY", "test")
    monkeypatch.setattr(planner, "gemma_rewrite", lambda text, key: ("Dal 99 plates", "gemma:test"))
    p = planner.make_plan(_result())
    assert p.source == "template" and "changed some numbers" in p.note


def test_gemma_network_error_falls_back(monkeypatch):
    monkeypatch.setenv("GEMMA_API_KEY", "test")

    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(planner.httpx, "post", boom)
    p = planner.make_plan(_result())
    assert p.source == "template" and "unreachable" in p.note


def test_gemma_request_shape(monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self): ...
        def json(self): return {"candidates": [{"content": {"parts": [{"text": "ok 12"}]}}]}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return Resp()

    monkeypatch.setattr(planner.httpx, "post", fake_post)
    text, source = planner.gemma_rewrite("Dal: 12 plates", "KEY")
    assert "gemma-3-27b-it:generateContent" in sent["url"] and sent["headers"]["x-goog-api-key"] == "KEY"
    assert "Dal: 12 plates" in sent["body"]["contents"][0]["parts"][0]["text"]
    assert text == "ok 12" and source == "gemma:gemma-3-27b-it"
