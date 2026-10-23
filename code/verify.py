"""Verify the replication outputs against the published manuscript tables.

Run after ``python code/run.py``:

    python code/verify.py

The script checks the source workbooks, the cleaned panel, the six tables, the
trend/break statistics and Figure 1.

Two kinds of tolerance are used, both documented in the README:

* Coefficients and test statistics are compared with the published values at
  the printed precision (three decimals).  The tolerance is therefore half a
  unit in the last printed digit (0.0005), i.e. the largest difference that
  correct rounding can produce.
* p-values get one extra unit in the last digit (0.001) because the published
  figures were rounded from intermediate quantities that carry optimizer and
  simulation noise of that order.  The OLS-GI bootstrap p-values in Table 6
  come from a separate Monte Carlo estimate and are checked with a 0.005
  tolerance (see the note printed by this script).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
MAN = ROOT / "manuscript" / "tables"

PRINTED = 0.0005   # half a unit in the third decimal
P_VALUE = 0.001    # printed p-values carry optimizer/simulation noise
BOOT_OLS = 0.005   # separate Monte Carlo estimate (Table 6, OLS-GI)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value) -> float:
    text = str(value).strip()
    if text.startswith("<"):
        return float(text[1:]) / 2.0
    return float(text)


def compare_table(
    table_no: int,
    columns: list[str],
    tolerances: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compare selected columns and return the worst difference per column."""
    ours = pd.read_csv(OUT / "tables" / f"table_{table_no}.csv")
    published = pd.read_csv(MAN / f"table_{table_no}.csv")
    assert len(ours) == len(published), (
        f"Table {table_no}: {len(ours)} rows vs {len(published)} published"
    )
    tolerances = tolerances or {}
    worst: dict[str, float] = {}
    for col in columns:
        tol = tolerances.get(col, PRINTED)
        diffs = [
            abs(number(a) - number(b)) for a, b in zip(ours[col], published[col])
        ]
        worst[col] = max(diffs)
        if worst[col] > tol + 1e-12:
            i = int(np.argmax(diffs))
            raise AssertionError(
                f"Table {table_no}, column {col}: replication {ours[col][i]} vs "
                f"published {published[col][i]} (diff {worst[col]:.6f} > {tol})"
            )
    return worst


def report(label: str, worst: dict[str, float]) -> None:
    detail = ", ".join(f"{col} {diff:.6f}" for col, diff in worst.items())
    print(f"  {label}: max |replication - published| -> {detail}")


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())

    # Source integrity.
    for rel, expected in manifest["source_workbooks"].items():
        actual = sha256(ROOT / rel)
        assert actual == expected, f"Source workbook changed: {rel}"

    # Cleaned panel.
    panel = pd.read_csv(OUT / "cleaned_panel.csv")
    assert len(panel) == 26000, f"expected 26,000 rows, found {len(panel)}"
    assert set(panel.fiscal_year) == set(range(1999, 2025))
    assert (panel.groupby("fiscal_year").size() == 1000).all()
    assert not panel[["fiscal_year", "rank"]].duplicated().any()
    assert (panel.groupby("fiscal_year")["rank"].max() == 1000).all()
    restored = panel.loc[(panel.fiscal_year == 2019) & (panel["rank"] == 779)].iloc[0]
    assert restored.company == "SG Comércio Exterior"
    assert restored.revenue_brl_million == 657.0
    assert manifest["panel"]["rows"] == 26000
    print("  panel: 26,000 rows, 1,000 firms per year, fiscal-2019 rank 779 restored")

    # Tables 1-6.
    p_cols = ["lmp_p_asym", "lmp_p_boot", "lmz_p_asym", "lmz_p_boot"]
    for table_no in (1, 2, 3):
        worst = compare_table(
            table_no,
            ["lmp", "lmz"] + p_cols,
            tolerances={c: P_VALUE for c in p_cols},
        )
        report(f"Table {table_no}", worst)

    worst = compare_table(
        4,
        ["years", "lmp_rejections", "alpha_mean", "alpha_min", "alpha_max"],
        tolerances={"years": 0.0, "lmp_rejections": 0.0},
    )
    report("Table 4", worst)

    worst = compare_table(
        5,
        ["pwl", "pwl_p_asym", "pwl_p_boot"],
        tolerances={"pwl_p_asym": P_VALUE, "pwl_p_boot": P_VALUE},
    )
    report("Table 5", worst)

    worst = compare_table(
        6,
        ["mle", "mle_p_asym", "mle_p_boot", "ols_gi", "ols_p_asym", "mle_abs_z"],
        tolerances={
            "mle_p_asym": P_VALUE,
            "mle_p_boot": P_VALUE,
            "ols_p_asym": P_VALUE,
        },
    )
    report("Table 6", worst)

    # The OLS-GI bootstrap p-values are a separate Monte Carlo estimate.
    ours6 = pd.read_csv(OUT / "tables" / "table_6.csv")
    pub6 = pd.read_csv(MAN / "table_6.csv")
    ols_diff = float(
        np.max(np.abs(ours6["ols_p_boot"].to_numpy() - pub6["ols_p_boot"].to_numpy()))
    )
    assert ols_diff <= BOOT_OLS, (
        f"OLS-GI bootstrap p-values differ by {ols_diff:.4f} "
        f"(tolerance {BOOT_OLS})"
    )
    print(
        f"  Table 6 OLS-GI bootstrap p-values: max |replication - published| = "
        f"{ols_diff:.4f} (Monte Carlo tolerance {BOOT_OLS})"
    )

    # Trend and break statistics.
    trend = pd.read_csv(OUT / "trend_break.csv").iloc[0]
    assert abs(trend.mk_s + 235.0) < 1e-9
    assert abs(trend.mk_z + 5.158) < 0.001
    assert abs(trend.mk_p - 2.5e-7) < 1e-9
    assert abs(trend.mk_tau + 0.723) < 0.001
    assert abs(trend.sen_slope_per_year + 0.014) < 0.0005
    assert abs(trend.sup_wald - 22.105) < 0.001
    assert int(trend.break_after) == 2009
    assert int(trend.regime_from) == 2010
    assert abs(trend.p_break_boot - 0.00015) < 1e-6
    assert abs(trend.pettitt_k - 165.0) < 1e-9
    assert abs(trend.pettitt_p - 0.000260) < 1e-6
    print("  trend/break: Mann-Kendall, Theil-Sen, sup-Wald, Pettitt and 2010 break")

    # Figure dimensions.
    import matplotlib.image as mpimg

    image = mpimg.imread(OUT / "figure1.png")
    assert image.shape[1] == 1440 and image.shape[0] == 840, image.shape
    print("  figure 1: 1440x840 px")

    print(
        "PASS: source hashes, 26,000-row panel, fiscal-2019 rank 779, "
        "Tables 1-6, trend/break statistics and Figure 1"
    )


if __name__ == "__main__":
    main()
