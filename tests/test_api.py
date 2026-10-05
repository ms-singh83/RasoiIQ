import pytest
from fastapi.testclient import TestClient

from app import forecast, main
from tests.conftest import make_csv


@pytest.fixture
def client(monkeypatch, fake_factory):
    monkeypatch.setattr(forecast, "make_tabpfn", fake_factory)
    monkeypatch.setattr(forecast.TabPFNForecaster.__init__, "__defaults__", (fake_factory,))
    main._cache.clear()
    with TestClient(main.app) as c:
        yield c


def test_page_and_config(client):
    assert "RasoiIQ" in client.get("/").text
    cfg = client.get("/api/config").json()
    assert cfg["friend"] and cfg["default_is_sample"] in (True, False)


def test_default_dataset_forecast(client):
    r = client.post("/api/forecast", data={"tomorrow_special": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_sample"] is True and body["items"] and body["backtest"]["tabpfn"]
    assert body["plan"]["source"] == "template"


def test_upload_forecast(client):
    r = client.post("/api/forecast", files={"file": ("mine.csv", make_csv(days=40), "text/csv")},
                    data={"tomorrow_special": "true"})
    body = r.json()
    assert r.status_code == 200 and body["is_sample"] is False and body["source_name"] == "mine.csv"
    assert body["festival"] and body["target_date"] == "2026-02-14"


def test_bad_upload_gives_friendly_400(client):
    r = client.post("/api/forecast", files={"file": ("bad.csv", "foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 400 and "Missing column" in r.json()["detail"]


def test_sample_download(client):
    r = client.get("/api/sample.csv")
    assert r.status_code == 200 and r.text.startswith("date,item,quantity")
