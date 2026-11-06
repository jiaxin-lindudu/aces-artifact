"""Experiment entry points; archived Ours scripts retain legacy import names.

Only extend the package lookup path. No archived experiment runs on import.
The maintained entry remains experiments.run_ours_experiments.
"""
from pathlib import Path as _Path

__path__.append(str(_Path(__file__).resolve().parent / 'legacy_ours'))
