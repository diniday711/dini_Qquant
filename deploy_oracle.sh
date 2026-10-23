#!/usr/bin/env bash
# ==============================================================================
# Q-Stack Oracle Cloud (Singapore Ubuntu/Linux) 1-Click Deployment Script
# Usage on Oracle Server:
#   git clone https://github.com/diniday711/dini_Qquant.git
#   cd dini_Qquant
#   cp .env.example .env && nano .env   # Paste your Binance Testnet API keys
#   bash deploy_oracle.sh
# ==============================================================================
set -e

echo "[Q-Stack] 1/4 Updating Python environment..."
python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas requests

echo "[Q-Stack] 2/4 Running W2 Risk & Leakage Unit Test Gate..."
python3 -m unittest tests/test_w2_and_leakage.py -v

echo "[Q-Stack] 3/4 Running Full 1-Cycle Pipeline (Regime Scan + Strategy Lab + Testnet Check + Dashboard)..."
python3 run_full_cycle.py

echo "[Q-Stack] 4/4 Complete! To run 24/7 daemon + web dashboard on port 8080:"
echo "  nohup .venv/bin/python3 run_full_cycle.py --daemon --port 8080 > qstack_daemon.log 2>&1 &"
