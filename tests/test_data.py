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
