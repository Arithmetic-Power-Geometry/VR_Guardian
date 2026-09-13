#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
pytest -q
python scripts/run_all.py
