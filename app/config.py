"""Who this is built for. Change these (or set the env vars) to personalise the page."""
import os
from pathlib import Path

FRIEND_NAME = os.getenv("FRIEND_NAME", "Simran")
CITY = os.getenv("CITY", "Chandigarh")
THEME = os.getenv("THEME", "Build for a Friend")

ROOT = Path(__file__).resolve().parent.parent
USER_ORDERS = ROOT / "orders.csv"
SAMPLE_ORDERS = ROOT / "data" / "sample_orders.csv"
SAMPLE_COMPLEX = ROOT / "data" / "sample_orders_complex.csv"
MODEL_CACHE = Path(os.getenv("TABPFN_MODEL_CACHE_DIR", ROOT / ".tabpfn_cache"))


def default_dataset() -> tuple[Path, bool]:
    """(path, is_sample). Her own orders.csv wins over the generated sample."""
    if USER_ORDERS.exists():
        return USER_ORDERS, False
    return SAMPLE_ORDERS, True
