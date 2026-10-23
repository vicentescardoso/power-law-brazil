"""Build the six manuscript tables, the trend/break statistics and Figure 1.

The functions in this module take the cleaned annual panel produced by
:mod:`dataload` and return plain :class:`pandas.DataFrame` objects.  No file
paths are hard-coded here; :mod:`run` writes the outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from plstats import (
    bootstrap_pvalue,
    lmz_test,
    lmp_test,
    pareto_mle,
    pwl_test,
    zipf_ols,
)

YEARS = tuple(range(1999, 2025))
CUTS = (1000, 500, 100)
GRID = (50, 75, 100, 125, 150, 200, 300, 500, 750, 999)

# Fixed seeds and replication counts used by the manuscript.
HIST_LMP_SEED = 20260701
HIST_LMZ_SEED = 20260702
HIST_REPS = 999
PWL_REPS = 19999
PWL_SEED_BASE = 20260923
ZIPF_BOOT_SEED = 20260908
ZIPF_BOOT_REPS = 20000
BREAK_REPS = 20000
TOP_N = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hill(x: np.ndarray, mu: float) -> tuple[float, float]:
    """Pareto type-I MLE (Hill) and its conditional standard error."""
    x = np.asarray(x, dtype=float)
    alpha = len(x) / np.log(x / mu).sum()
    return float(alpha), float(alpha / np.sqrt(len(x)))


def sup_wald(y: np.ndarray, se: np.ndarray, minseg: int = 4):
    """Inverse-variance weighted one-step mean-change statistic.

    Returns ``(stat, split, mean_left, mean_right)`` where ``split`` is the
    number of annual observations in the first segment.
    """
    t = len(y)
    w = 1.0 / np.square(se)
    best = (-1.0, -1, np.nan, np.nan)
    for split in range(minseg, t - minseg + 1):
        w1, w2 = w[:split], w[split:]
        m1 = np.average(y[:split], weights=w1)
        m2 = np.average(y[split:], weights=w2)
        stat = (m1 - m2) ** 2 / (1.0 / w1.sum() + 1.0 / w2.sum())
        if stat > best[0]:
            best = (float(stat), split, float(m1), float(m2))
    return best


def mann_kendall(y: np.ndarray) -> tuple[float, float, float, float]:
    """Mann-Kendall S, Kendall tau, continuity-corrected Z and two-sided p."""
    y = np.asarray(y, dtype=float)
    t = len(y)
    s = 0.0
    for i in range(t - 1):
        for j in range(i + 1, t):
            s += np.sign(y[j] - y[i])
    _, counts = np.unique(y, return_counts=True)
    tie = sum(int(c) * (int(c) - 1) * (2 * int(c) + 5) for c in counts if c > 1)
    var = (t * (t - 1) * (2 * t + 5) - tie) / 18.0
    if s > 0:
        z = (s - 1.0) / np.sqrt(var)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var)
    else:
        z = 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    tau = s / (t * (t - 1) / 2.0)
    return float(s), float(tau), float(z), float(p)


def pettitt(y: np.ndarray) -> tuple[float, int, float]:
    """Pettitt rank change-point statistic, split index and approximate p."""
    y = np.asarray(y, dtype=float)
    t = len(y)
    u = np.array(
        [
            np.sum(
                [
                    np.sign(y[i] - y[j])
                    for i in range(split + 1)
                    for j in range(split + 1, t)
                ]
            )
            for split in range(t - 1)
        ]
    )
    k = float(np.max(np.abs(u)))
    split = int(np.argmax(np.abs(u)))
    p = 2.0 * np.exp(-6.0 * k * k / (t ** 3 + t ** 2))
    return k, split, float(p)


# ---------------------------------------------------------------------------
# Tables 1-3: LMp and LMz score tests at the three cutoffs
# ---------------------------------------------------------------------------

def historical_bootstrap(panel_old: dict[int, pd.DataFrame], reps: int = HIST_REPS):
    """Replicate the 999-replication calibration reported in Tables 1-3.

    The calibration uses the 999-firm fiscal-2019 sample (without the
    recovered rank-779 observation) and the two fixed streams documented in
    the manuscript: seed 20260701 for LMp/PWL and seed 20260702 for LMz.
    The PWL draws are consumed even though Tables 1-3 do not report them, so
    that the LMp stream advances exactly as in the original calibration.
    """
    rng_lmp = np.random.default_rng(HIST_LMP_SEED)
    rng_lmz = np.random.default_rng(HIST_LMZ_SEED)
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for fy in sorted(panel_old):
        sales = panel_old[fy]["sales"].to_numpy(dtype=float)
        for n in (100, 500, 1000):
            x = sales[:n] if len(sales) >= n else sales.copy()
            p_lmp = bootstrap_pvalue(x, lmp_test, reps=reps, rng=rng_lmp)
            p_lmz = bootstrap_pvalue(
                x, lmz_test, reps=reps, rng=rng_lmz, zipf_null=True
            )
            below = sales[sales < x.min()]
            mu_pwl = below.max() if below.size else x.min() * 0.999
            bootstrap_pvalue(x, pwl_test, mu=float(mu_pwl), reps=reps, rng=rng_lmp)
            out[(fy, n)] = (float(p_lmp), float(p_lmz))
    return out


def build_tables_1_3(
    panel: dict[int, pd.DataFrame],
    panel_old: dict[int, pd.DataFrame],
) -> dict[int, pd.DataFrame]:
    """Return ``{n: DataFrame}`` for the top-1000, top-500 and top-100 tables."""
    boot = historical_bootstrap(panel_old)
    out: dict[int, pd.DataFrame] = {}
    for n in CUTS:
        rows = []
        for fy in YEARS:
            x = panel[fy]["sales"].to_numpy(dtype=float)[:n]
            mu = float(x.min())
            alpha, se = hill(x, mu)
            lmp, p_lmp = lmp_test(x, mu)
            lmz, p_lmz = lmz_test(x, mu)
            p_lmp_b, p_lmz_b = boot[(fy, n)]
            rows.append(
                {
                    "fiscal_year": fy,
                    "n": n,
                    "alpha_mle": alpha,
                    "se_mle": se,
                    "lmp": float(lmp),
                    "lmp_p_asym": float(p_lmp),
                    "lmp_p_boot": p_lmp_b,
                    "lmz": float(lmz),
                    "lmz_p_asym": float(p_lmz),
                    "lmz_p_boot": p_lmz_b,
                }
            )
        out[n] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Table 4: cutoff grid
# ---------------------------------------------------------------------------

def build_table4(panel: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """LMp rejection counts and Hill exponents over the cutoff grid."""
    rows = []
    for n in GRID:
        alphas = []
        rejects = 0
        for fy in YEARS:
            sales = panel[fy]["sales"].to_numpy(dtype=float)
            x = sales[:n]
            cutoff = float(sales[n])  # revenue of rank n+1
            alpha, _ = hill(x, cutoff)
            _, p_lmp = lmp_test(x, cutoff)
            alphas.append(alpha)
            rejects += int(p_lmp < 0.05)
        rows.append(
            {
                "n": n,
                "years": len(YEARS),
                "lmp_rejections": rejects,
                "alpha_mean": float(np.mean(alphas)),
                "alpha_min": float(np.min(alphas)),
                "alpha_max": float(np.max(alphas)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 5: PWL test at the top-100 cutoff
# ---------------------------------------------------------------------------

def build_table5(panel: dict[int, pd.DataFrame], reps: int = PWL_REPS) -> pd.DataFrame:
    """PWL score test of Pareto I against Pareto IV, top 100 firms."""
    rows = []
    for fy in YEARS:
        sales = panel[fy]["sales"].to_numpy(dtype=float)
        x = sales[:TOP_N]
        cutoff = float(sales[TOP_N])  # rank 101
        stat, p_asym = pwl_test(x, cutoff)
        seed = PWL_SEED_BASE + fy
        p_boot = bootstrap_pvalue(
            x,
            pwl_test,
            mu=cutoff,
            reps=reps,
            rng=np.random.default_rng(seed),
        )
        rows.append(
            {
                "fiscal_year": fy,
                "n": TOP_N,
                "cutoff_rank": TOP_N + 1,
                "cutoff": cutoff,
                "pwl": float(stat),
                "pwl_p_asym": float(p_asym),
                "pwl_p_boot": float(p_boot),
                "reps": reps,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 6: Hill and Gabaix-Ibragimov estimates and Zipf tests
# ---------------------------------------------------------------------------

def build_table6(
    panel: dict[int, pd.DataFrame],
    rng: np.random.Generator,
    reps: int = ZIPF_BOOT_REPS,
) -> pd.DataFrame:
    """Top-100 Hill and OLS-GI estimates, asymptotic and bootstrap p-values.

    The MLE bootstrap simulates ``reps`` Pareto type-I samples of size 100
    with ``alpha = 1`` and re-estimates the Hill exponent.  The OLS-GI
    bootstrap uses the same uniform draws, transforms them to Pareto type-I
    samples with ``alpha = 1``, and re-estimates the rank-size exponent.
    """
    u = rng.random((reps, TOP_N))

    # Hill bootstrap under alpha = 1.
    theta_b = TOP_N / np.sum(-np.log(u), axis=1)
    se_mle_b = theta_b / np.sqrt(TOP_N)
    t_mle_b = np.abs(theta_b - 1.0) / se_mle_b

    # OLS-GI bootstrap on the same draws.
    x_b = np.sort(u ** (-1.0), axis=1)[:, ::-1]
    log_x = np.log(x_b)
    log_rank = np.log(np.arange(1, TOP_N + 1) - 0.5)
    log_rank_c = log_rank - log_rank.mean()
    log_x_c = log_x - log_x.mean(axis=1, keepdims=True)
    slope_b = (log_x_c * log_rank_c).sum(axis=1) / (log_x_c ** 2).sum(axis=1)
    alpha_ols_b = -slope_b
    se_ols_b = alpha_ols_b * np.sqrt(2.0 / TOP_N)
    t_ols_b = np.abs(alpha_ols_b - 1.0) / se_ols_b

    rows = []
    for fy in YEARS:
        x = panel[fy]["sales"].to_numpy(dtype=float)[:TOP_N]
        theta, se_mle = hill(x, float(x.min()))
        z_mle = abs(theta - 1.0) / se_mle
        p_mle_asym = 2.0 * stats.norm.sf(z_mle)
        p_mle_boot = (1.0 + np.sum(t_mle_b >= z_mle)) / (reps + 1.0)

        slope, _, _, se_ols = zipf_ols(x)
        alpha_ols = -slope
        z_ols = abs(alpha_ols - 1.0) / se_ols
        p_ols_asym = 2.0 * stats.norm.sf(z_ols)
        p_ols_boot = (1.0 + np.sum(t_ols_b >= z_ols)) / (reps + 1.0)

        rows.append(
            {
                "fiscal_year": fy,
                "mle": float(theta),
                "mle_se": float(se_mle),
                "mle_p_asym": float(p_mle_asym),
                "mle_p_boot": float(p_mle_boot),
                "ols_gi": float(alpha_ols),
                "ols_gi_se": float(se_ols),
                "ols_p_asym": float(p_ols_asym),
                "ols_p_boot": float(p_ols_boot),
                "mle_abs_z": float(z_mle),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trend and structural break in the top-100 Hill exponent
# ---------------------------------------------------------------------------

def top100_series(panel: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = np.array(sorted(panel), dtype=int)
    theta = []
    se = []
    for fy in years:
        x = panel[fy]["sales"].to_numpy(dtype=float)[:TOP_N]
        alpha, alpha_se = hill(x, float(x.min()))
        theta.append(alpha)
        se.append(alpha_se)
    return years, np.asarray(theta), np.asarray(se)


def build_trend_break(
    panel: dict[int, pd.DataFrame],
    rng: np.random.Generator,
    reps: int = BREAK_REPS,
) -> pd.DataFrame:
    """Mann-Kendall, Theil-Sen, weighted sup-Wald and Pettitt statistics."""
    years, y, se = top100_series(panel)
    t = len(y)
    s, tau, z, p_mk = mann_kendall(y)
    sen = stats.theilslopes(y, years)

    w = 1.0 / se ** 2
    mu0 = float(np.sum(w * y) / np.sum(w))
    sup, split, mean_left, mean_right = sup_wald(y, se)
    k_values = range(4, t - 4)

    # Heteroskedastic parametric bootstrap under the no-break null.
    errors = rng.normal(0.0, 1.0, (reps, t)) * se[None, :]
    draws = mu0 + errors
    sup_b = np.empty(reps)
    for b in range(reps):
        yb = draws[b]
        best = -1.0
        for k in k_values:
            w1, w2 = w[:k], w[k:]
            m1 = np.sum(w1 * yb[:k]) / w1.sum()
            m2 = np.sum(w2 * yb[k:]) / w2.sum()
            stat = (m1 - m2) ** 2 / (1.0 / w1.sum() + 1.0 / w2.sum())
            if stat > best:
                best = stat
        sup_b[b] = best
    p_break = (1.0 + np.sum(sup_b >= sup)) / (reps + 1.0)

    k_pet, split_pet, p_pet = pettitt(y)

    return pd.DataFrame(
        [
            {
                "sample": f"Top {TOP_N}",
                "n_years": t,
                "mk_s": s,
                "mk_tau": tau,
                "mk_z": z,
                "mk_p": p_mk,
                "sen_slope_per_year": float(sen.slope),
                "sup_wald": float(sup),
                "break_after": int(years[split - 1]),
                "regime_from": int(years[split]),
                "mean_pre": float(mean_left),
                "mean_post": float(mean_right),
                "mean_pre_unweighted": float(y[:split].mean()),
                "mean_post_unweighted": float(y[split:].mean()),
                "p_break_boot": float(p_break),
                "break_reps": reps,
                "pettitt_k": k_pet,
                "pettitt_after": int(years[split_pet]),
                "pettitt_p": p_pet,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def build_figure(panel: dict[int, pd.DataFrame], png_path, pdf_path) -> None:
    """Black-and-white top-100 Hill exponent figure with trend and break."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    years, y, se = top100_series(panel)
    sen = stats.theilslopes(y, years)
    _, split, _, _ = sup_wald(y, se)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(
        years,
        y - 1.96 * se,
        y + 1.96 * se,
        color="0.82",
        edgecolor="none",
        label="95% band (MLE ± 1.96·SE)",
    )
    ax.plot(
        years,
        y,
        color="black",
        marker="o",
        ms=3.5,
        lw=1.1,
        label="θ̂ MLE top-100",
    )
    xx = np.array([years[0], years[-1]])
    ax.plot(
        xx,
        np.median(y) + sen.slope * (xx - np.median(years)),
        color="black",
        ls="--",
        lw=1.1,
        label=f"Theil–Sen ({sen.slope:+.4f}/yr)",
    )
    ax.axvline(years[split] - 0.5, color="black", ls=":", lw=1.1)
    ax.text(
        years[split] - 0.4,
        y.max() - 0.02,
        f"break: regime from {years[split]}",
        fontsize=8,
        va="top",
    )
    ax.hlines(
        [y[:split].mean(), y[split:].mean()],
        [years[0], years[split]],
        [years[split - 1], years[-1]],
        colors="black",
        linestyles="dashdot",
        lw=1.0,
    )
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Pareto exponent θ (top 100)")
    ax.set_title("Tail exponent, top-100 firms (B&W)")
    ax.legend(frameon=True, edgecolor="black", fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
