#!/bin/bash
set -e
cd "$(dirname "$0")"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; NC='\033[0m'

echo -e "${C}=================================================${NC}"
echo -e "${C}  Gold Quant — Run 3 Strategy Backtests${NC}"
echo -e "${C}=================================================${NC}"

# ── Step 1: Python venv + dependencies ──────────────────────
if [ ! -d ".venv" ]; then
    echo -e "\n${Y}[1/4] Creating Python venv...${NC}"
    python3 -m venv .venv
else
    echo -e "\n${G}[1/4] venv already exists${NC}"
fi

source .venv/bin/activate
echo "  Python: $(python3 --version)"

if ! python3 -c "import numpy" 2>/dev/null; then
    echo -e "${Y}  Installing numpy, pandas, scipy...${NC}"
    pip install --upgrade pip -q
    pip install numpy pandas scipy -q
    echo -e "${G}  Dependencies installed${NC}"
else
    echo -e "${G}  Dependencies already installed${NC}"
fi

# ── Step 2: Merge M15 data if needed ───────────────────────
M15_TARGET="data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv"
if [ ! -f "$M15_TARGET" ]; then
    echo -e "\n${Y}[2/4] Merging M15 data files...${NC}"
    python3 -c "
import glob, os
files = sorted(glob.glob('data/download/download/xauusd-m15-bid-*.csv'))
if not files:
    print('ERROR: No M15 source files found'); exit(1)
with open('$M15_TARGET', 'w') as out:
    for i, f in enumerate(files):
        with open(f) as inp:
            for j, line in enumerate(inp):
                if j == 0 and i > 0:
                    continue
                out.write(line)
print(f'  Merged {len(files)} files -> $M15_TARGET')
"
else
    echo -e "\n${G}[2/4] M15 data already merged${NC}"
fi

# ── Step 3: Run 3 strategies sequentially ──────────────────
echo -e "\n${C}=================================================${NC}"
echo -e "${C}  [3/4] Running Strategy A: Momentum Chase${NC}"
echo -e "${C}=================================================${NC}"
START_A=$(date +%s)
python3 run_strategy_a_momentum.py 2>&1
END_A=$(date +%s)
echo -e "${G}  Strategy A done in $((END_A - START_A))s${NC}"

echo -e "\n${C}=================================================${NC}"
echo -e "${C}  [3/4] Running Strategy C: Multi-TF Trend Filter${NC}"
echo -e "${C}=================================================${NC}"
START_C=$(date +%s)
python3 run_strategy_c_trend_filter.py 2>&1
END_C=$(date +%s)
echo -e "${G}  Strategy C done in $((END_C - START_C))s${NC}"

echo -e "\n${C}=================================================${NC}"
echo -e "${C}  [3/4] Running Strategy D: Trend Pullback${NC}"
echo -e "${C}=================================================${NC}"
START_D=$(date +%s)
python3 run_strategy_d_pullback.py 2>&1
END_D=$(date +%s)
echo -e "${G}  Strategy D done in $((END_D - START_D))s${NC}"

# ── Step 4: Summary ───────────────────────────────────────
echo -e "\n${C}=================================================${NC}"
echo -e "${C}  [4/4] All Done!${NC}"
echo -e "${C}=================================================${NC}"
TOTAL=$((END_D - START_A))
echo -e "  Strategy A: $((END_A - START_A))s"
echo -e "  Strategy C: $((END_C - START_C))s"
echo -e "  Strategy D: $((END_D - START_D))s"
echo -e "  Total: ${TOTAL}s ($(( TOTAL / 60 ))min)"
echo ""
echo -e "  Output files:"
echo -e "    ${G}strategy_a_output.txt${NC}"
echo -e "    ${G}strategy_c_output.txt${NC}"
echo -e "    ${G}strategy_d_output.txt${NC}"
