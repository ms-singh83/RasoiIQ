"""Read and clean an orders CSV (date, item, quantity, optional notes)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .festivals import note_marks_festival

COLUMN_ALIASES = {
    "date": ("date", "day", "order_date"),
    "item": ("item", "dish", "product", "item_name"),
    "quantity": ("quantity", "qty", "orders", "plates", "count"),
    "notes": ("notes", "note", "remarks", "comment"),
}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MIN_DAYS = 21
# TabPFN on CPU is happiest under 1,000 training rows (days x items).
MAX_ROWS = 1000


class DataError(ValueError):
    """A problem with the CSV, worded for the person who uploaded it."""


@dataclass
class OrdersData:
    daily: pd.DataFrame  # date, item, quantity on a full date x item grid
    items: list[str]
    start: date
    end: date
    special_dates: set[date] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


def _match_columns(df: pd.DataFrame) -> dict[str, str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    found = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                found[canon] = lower[a]
                break
    missing = [c for c in ("date", "item", "quantity") if c not in found]
    if missing:
        raise DataError(f"Missing column(s): {', '.join(missing)}. The header should be: date,item,quantity")
    return found


def _parse_dates(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if iso.notna().mean() >= 0.9:
        return iso
    return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")  # 05/10/2026 = 5 Oct


def load_orders(raw: bytes | str) -> OrdersData:
    if isinstance(raw, bytes):
        if len(raw) > MAX_UPLOAD_BYTES:
            raise DataError("That file is over 2 MB. Please upload a smaller CSV.")
        raw = raw.decode("utf-8-sig", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(raw), dtype=str, skipinitialspace=True, keep_default_na=False)
    except Exception as exc:
        raise DataError(f"Could not read the CSV: {exc}") from exc
    if df.empty:
        raise DataError("The CSV has no rows.")

    cols = _match_columns(df)
    out = pd.DataFrame({
        "date": _parse_dates(df[cols["date"]].str.strip()),
        "item": df[cols["item"]].str.strip().str.title(),
        "quantity": pd.to_numeric(df[cols["quantity"]].str.strip(), errors="coerce"),
        "notes": df[cols["notes"]] if "notes" in cols else "",
    })
    warnings: list[str] = []
    bad = out["date"].isna() | (out["item"] == "") | out["quantity"].isna() | (out["quantity"] < 0)
    if bad.any():
        warnings.append(f"Skipped {int(bad.sum())} row(s) with a missing or invalid date, item or quantity.")
    out = out[~bad]
    if out.empty:
        raise DataError("No valid rows found. Check the date, item and quantity columns.")
    out = out.assign(date=out["date"].dt.normalize())

    special = {d.date() for d, n in zip(out["date"], out["notes"]) if note_marks_festival(n)}
    agg = out.groupby(["date", "item"], as_index=False)["quantity"].sum()
    items = sorted(agg["item"].unique())
    start, end = agg["date"].min(), agg["date"].max()

    max_days = MAX_ROWS // len(items)
    if (end - start).days + 1 > max_days:
        start = end - pd.Timedelta(days=max_days - 1)
        agg = agg[agg["date"] >= start]
        warnings.append(
            f"Using the most recent {max_days} days so the table stays under {MAX_ROWS} rows for TabPFN on CPU."
        )
    n_days = (end - start).days + 1
    if n_days < MIN_DAYS:
        raise DataError(f"Only {n_days} days of orders. RasoiIQ needs at least {MIN_DAYS} days.")

    grid = pd.MultiIndex.from_product([pd.date_range(start, end, freq="D"), items], names=["date", "item"])
    daily = agg.set_index(["date", "item"]).reindex(grid, fill_value=0.0).reset_index()
    missing_days = n_days - agg["date"].nunique()
    if missing_days:
        warnings.append(f"{missing_days} day(s) had no orders at all; counted as 0.")
    return OrdersData(daily=daily, items=items, start=start.date(), end=end.date(),
                      special_dates=special, warnings=warnings)
