import pytest

from app.data import MAX_ROWS, DataError, load_orders
from tests.conftest import make_csv


def test_loads_and_fills_grid():
    d = load_orders(make_csv(days=30))
    assert d.items == ["Dal", "Rajma"]
    assert len(d.daily) == 60
    assert str(d.start) == "2026-01-05" and str(d.end) == "2026-02-03"


def test_column_aliases_daymonth_dates_and_duplicates_summed():
    rows = ["Date,Dish,Qty"]
    for day in range(1, 29):
        rows.append(f"{day:02d}/02/2026,dal makhani,5")
    rows.append("03/02/2026,dal makhani,4")  # second order the same day
    d = load_orders("\n".join(rows))
    assert d.items == ["Dal Makhani"]
    assert str(d.start) == "2026-02-01"  # 01/02 read as 1 Feb, not 2 Jan
    assert d.daily.set_index("date").loc["2026-02-03", "quantity"] == 9


def test_bad_rows_are_skipped_with_warning():
    csv = make_csv(days=25) + "not-a-date,Dal,3\n2026-01-06,Dal,abc\n2026-01-07,,4\n2026-01-08,Dal,-1\n"
    d = load_orders(csv)
    assert any("Skipped 4" in w for w in d.warnings)


def test_missing_column():
    with pytest.raises(DataError, match="quantity"):
        load_orders("date,item\n2026-01-01,Dal\n")


def test_too_few_days():
    with pytest.raises(DataError, match="at least"):
        load_orders(make_csv(days=10))


def test_row_cap_keeps_most_recent_days():
    d = load_orders(make_csv(days=400, items=("A", "B", "C")))
    assert len(d.daily) <= MAX_ROWS
    assert str(d.end) == "2027-02-08"
    assert any("most recent" in w for w in d.warnings)


def test_notes_mark_special_days():
    csv = make_csv(days=25) + "2026-01-20,Dal,30,big party order\n"
    d = load_orders(csv)
    assert any(str(x) == "2026-01-20" for x in d.special_dates)


def test_sample_file_is_under_row_limit():
    from app.config import SAMPLE_ORDERS
    d = load_orders(SAMPLE_ORDERS.read_bytes())
    assert len(d.daily) < MAX_ROWS and (d.end - d.start).days + 1 == 90


def test_all_zero_day_is_treated_as_closed():
    csv = make_csv(days=30).replace("2026-01-20,Dal,", "2026-01-20,Dal,0#").replace("2026-01-20,Rajma,", "2026-01-20,Rajma,0#")
    csv = "\n".join(l.split("#")[0] + ("," if "#" in l else "") for l in csv.splitlines())
    d = load_orders(csv)
    day = d.daily[d.daily["date"] == "2026-01-20"]
    assert day["quantity"].isna().all()
    assert any("kitchen closed" in w for w in d.warnings)
    # a single dish at zero on an open day is a real zero
    csv2 = make_csv(days=30).replace("2026-01-21,Dal,", "2026-01-21,Dal,0#")
    csv2 = "\n".join(l.split("#")[0] + ("," if "#" in l else "") for l in csv2.splitlines())
    d2 = load_orders(csv2)
    assert d2.daily.set_index(["date", "item"]).loc[("2026-01-21", "Dal"), "quantity"] == 0


def test_closed_day_does_not_zero_out_features():
    from app.features import build_features
    rows = make_csv(days=40).splitlines()
    rows = [r for r in rows if not r.startswith("2026-01-30")]  # nothing recorded that day
    d = load_orders("\n".join(rows))
    f = build_features(d.daily, d.items)
    nxt = f[(f["date"] == "2026-01-31") & (f["item"] == "Dal")].iloc[0]
    assert nxt["mean_7"] > 5  # average ignores the closed day instead of counting a 0


def test_complex_sample_order_log():
    """The messy per-order sample: aliases, day-first dates, spellings, bad rows, closed days."""
    from app.config import ROOT
    d = load_orders((ROOT / "data" / "sample_orders_complex.csv").read_bytes())
    assert d.items == ["Aloo Paratha", "Chole Bhature", "Dal Makhani", "Kadhi Chawal",
                       "Paneer Butter Masala", "Rajma Chawal"]
    assert str(d.start) == "2026-07-22" and str(d.end) == "2026-10-04"
    assert len(d.daily) == 75 * 6 < MAX_ROWS
    assert any("Skipped 4" in w for w in d.warnings)
    assert any("2 day(s)" in w and "closed" in w for w in d.warnings)
    chole = d.daily[(d.daily["item"] == "Chole Bhature") & (d.daily["date"] < "2026-09-01")]
    assert (chole["quantity"].fillna(0) == 0).all()  # not on the menu yet
