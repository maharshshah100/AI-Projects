#!/usr/bin/env bash
set -e
echo "=== FinSight setup ==="
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo; echo "Setup complete. Now run:  ./run.sh   (optional: add API keys to .env)"
