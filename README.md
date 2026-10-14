# ELTL++ Complete Validation Package
# Includes: Data Quality, Contracts, Time Travel, Versioning, Performance, Cost Analysis

## Overview

This package validates ELTL++ with focus on:
1. **Data Quality & Contracts** (explicit, executable validation rules)
2. **Time Travel & Versioning** (using Delta Lake)
3. **Query Performance** (Raw L1 vs Semantic L2)
4. **Cost Analysis** (Corrected with realistic storage pricing)

**Prerequisites:**
Use Python 3.11 and the exact package versions in `requirements.txt`.

**Reference runtime:** under one minute on the documented local environment.
**Cost:** < $0 (runs locally with DuckDB + local Delta Lake)

---

## File Structure

```
eltl_validation/
├── run_all_experiments.py          # Main orchestrator
├── experiments/
│   ├── 1_data_generation.py        # Generate TPC-H dataset
│   ├── 2_data_quality.py           # Data quality & contracts
│   ├── 3_time_travel.py            # Versioning & time travel
│   ├── 4_performance.py            # Query performance
│   └── 5_cost_analysis.py          # Corrected cost model
├── data/                            # Generated at runtime (untracked)
├── results/                         # Generated at runtime (untracked)
└── requirements.txt
```

---

## Installation & Usage

```bash
# 1. Clone the versioned artifact
git clone https://github.com/ChiaraRucco/ETLT_DesignPattern_validation.git
cd ETLT_DesignPattern_validation

# 2. Install the pinned reference environment
pip install -r requirements.txt

# 3. Run the complete validation (no interactive input required)
python run_all_experiments.py
```

---

## Reproducibility safeguards

The generator uses fixed random seeds and a fixed reference date. Each execution
deletes and rebuilds `data/`, runs the performance test before versioning mutates
the raw table, and writes the Python/platform metadata to `results/environment.json`.
Generated data and results are excluded from version control so a rerun starts from
a clean state.
`VALIDATION_REPORT.txt` derives its performance and cost summary from the JSON
result files; it does not contain hard-coded outcome claims. Query timing remains
hardware-dependent and should be interpreted as a local benchmark, not a universal
speedup guarantee.

---

## Files Generated

```
results/
├── data_quality_results.json       # Contract validation details
├── performance_results.json        # Query benchmark results
├── cost_analysis_results.json      # Cost comparison data
└── VALIDATION_REPORT.txt           # Final consolidated report
```

---

## What This Validates

✅ **Data Quality:** executable schema and quality-rule validation
✅ **Time Travel:** Delta Lake versioning with full history
✅ **Performance:** a controlled Raw L1 versus Semantic L2 comparison
✅ **Cost:** a transparent, parameterized tiered-storage model
✅ **Reproducibility:** a pinned, end-to-end local validation workflow

This package provides a reproducible implementation artifact. The generated
measurements are limited to its stated workload and local execution environment.
