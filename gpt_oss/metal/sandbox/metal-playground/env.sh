#!/usr/bin/env bash
# Optional helper to set up your venv quickly.
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Env ready. Run: source .venv/bin/activate"
