"""RasoiIQ: tomorrow's order forecast for a home kitchen, powered by TabPFN."""
import os

# Privacy and offline use: never phone home from the TabPFN package.
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
