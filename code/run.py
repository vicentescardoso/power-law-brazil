"""Run the full replication pipeline.

Usage
-----
    python code/run.py

The script reads the three workbooks in ``data/raw`` and writes every table,
the figure and the cleaned panel to ``output``.  All random draws use fixed
seeds, so repeated runs produce identical CSV files.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dataload import (
    ANNUAL_2020_XLSX,
    HIST_XLSX,
    PAINEL_XLSX,
    load_panel,
    qa_panel,
    without_2019_restoration,
)
from tables import (
    BREAK_REPS,
    HIST_LMZ_SEED,
    HIST_LMP_SEED,
    HIST_REPS,
    PWL_REPS,
    PWL_SEED_BASE,
    ZIPF_BOOT_REPS,
    ZIPF_BOOT_SEED,
    build_figure,
    build_table4,
    build_table5,
    build_table6,
    build_tables_1_3,
    build_trend_break,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
TABLES = OUT / "tables"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_cleaned_panel(panel: dict[int, pd.DataFrame]) -> int:
    rows = []
    for fy in sorted(panel):
        frame = panel[fy]
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            rows.append(
                {
                    "fiscal_year": fy,
                    "rank": rank,
                    "company": row.company,
                    "company_clean": row.clean,
                    "revenue_brl_million": float(row.sales),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cleaned_panel.csv", index=False)
    return len(df)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    panel = load_panel(verbose=False, restore_2019=True)
    issues = qa_panel(panel, verbose=False)
    if issues:
        raise RuntimeError("Source panel failed QA: " + "; ".join(issues))

    rows = write_cleaned_panel(panel)
    panel_old = without_2019_restoration(panel)

    # Tables 1-3.
    t13 = build_tables_1_3(panel, panel_old)
    for table_no, n in ((1, 1000), (2, 500), (3, 100)):
        t13[n].to_csv(TABLES / f"table_{table_no}.csv", index=False)

    # Table 4.
    build_table4(panel).to_csv(TABLES / "table_4.csv", index=False)

    # Table 5.
    build_table5(panel, reps=PWL_REPS).to_csv(TABLES / "table_5.csv", index=False)

    # Table 6 and the trend/break statistics share one RNG stream, so that
    # the break bootstrap follows the same 20,000 Pareto draws used by the
    # Zipf bootstrap (the order used in the original calibration).
    rng = np.random.default_rng(ZIPF_BOOT_SEED)
    table6 = build_table6(panel, rng, reps=ZIPF_BOOT_REPS)
    table6.to_csv(TABLES / "table_6.csv", index=False)
    build_trend_break(panel, rng, reps=BREAK_REPS).to_csv(
        OUT / "trend_break.csv", index=False
    )

    # Figure 1.
    build_figure(panel, OUT / "figure1.png", OUT / "figure1.pdf")

    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "source_workbooks": {
            str(PAINEL_XLSX.relative_to(ROOT)): sha256(PAINEL_XLSX),
            str(HIST_XLSX.relative_to(ROOT)): sha256(HIST_XLSX),
            str(ANNUAL_2020_XLSX.relative_to(ROOT)): sha256(ANNUAL_2020_XLSX),
        },
        "panel": {
            "first_year": min(panel),
            "last_year": max(panel),
            "firms_per_year": 1000,
            "rows": rows,
        },
        "repairs": {
            "fiscal_2012_2014": "rebuilt from historical workbook sheets 2013, 2014, 2015",
            "fiscal_2019_rank_779": "SG Comércio Exterior, MG, R$ 657.0 million, recovered from the 2020 annual edition",
        },
        "seeds": {
            "historical_lmp": HIST_LMP_SEED,
            "historical_lmz": HIST_LMZ_SEED,
            "pwl_base": PWL_SEED_BASE,
            "zipf_bootstrap": ZIPF_BOOT_SEED,
        },
        "replications": {
            "historical_tests": HIST_REPS,
            "pwl": PWL_REPS,
            "zipf_bootstrap": ZIPF_BOOT_REPS,
            "break_bootstrap": BREAK_REPS,
        },
        "outputs": [
            "cleaned_panel.csv",
            "tables/table_1.csv",
            "tables/table_2.csv",
            "tables/table_3.csv",
            "tables/table_4.csv",
            "tables/table_5.csv",
            "tables/table_6.csv",
            "trend_break.csv",
            "figure1.png",
            "figure1.pdf",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "rows": rows,
                "qa_issues": issues,
                "tables": {
                    "top1000": len(t13[1000]),
                    "top500": len(t13[500]),
                    "top100": len(t13[100]),
                    "cutoff_grid": 10,
                    "pwl": 26,
                    "zipf": len(table6),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
