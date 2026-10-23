"""
plstats.py — Estimators and score tests for power-law (Pareto) behavior.

Implements, in pure Python/NumPy/SciPy:
  * Pareto type I maximum likelihood (Hill) estimator;
  * Gabaix–Ibragimov (2011) rank-1/2 OLS Zipf regression;
  * Lomax / Pareto type II maximum likelihood (profiled);
  * Goerlich (2013) LMP score test of Pareto I vs Pareto II;
  * Urzúa (2000) LMZ score test of Zipf vs Pareto II;
  * Urzúa (2020) PWL score test of Pareto I vs Pareto IV
    (previously available only as the Stata module `pwlaw`);
  * parametric-bootstrap finite-sample p-values for the three tests;
  * Vuong (1989) non-nested test of Pareto I vs truncated lognormal;
  * concentration measures (CRk, HHI, Gini).

Conventions follow the source articles.  For LMP/LMZ the location mu is the
sample minimum (as in Stata's paretofit; the observation at the minimum
contributes log(x/mu)=0).  For PWL the density requires x > mu, so mu must be
strictly below the smallest observation; pass the next-smaller value in the
parent sample (Urzúa 2020, Table 2 note).
"""
import numpy as np
from scipy import stats
from scipy.special import digamma, polygamma


# ── Estimators ────────────────────────────────────────────────────────────────

def pareto_mle(x, mu=None):
    """Pareto I MLE. Returns (theta_hat, se, mu, n)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if mu is None:
        mu = x.min()
    theta = n / np.sum(np.log(x / mu))
    return theta, theta / np.sqrt(n), mu, n


def zipf_ols(x):
    """Gabaix–Ibragimov log(rank − 1/2) on log(size) regression.

    Returns (slope, intercept, r2, se_GI) where se_GI = |slope|·sqrt(2/n).
    """
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    n = x.size
    lr = np.log(np.arange(1, n + 1) - 0.5)
    ls = np.log(x)
    slope, intercept, r, _, _ = stats.linregress(ls, lr)
    return slope, intercept, r ** 2, abs(slope) * np.sqrt(2.0 / n)


def lomax_mle(x, mu=None):
    """Pareto II (Lomax with location mu) MLE via profiled likelihood.

    f(x) = (theta/sigma) (1 + (x-mu)/sigma)^-(theta+1),  x > mu.
    Returns (theta_hat, sigma_hat, loglik).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if mu is None:
        mu = x.min()
    w = x - mu

    def negprof(log_sig):
        sigma = np.exp(log_sig)
        S = np.sum(np.log1p(w / sigma))
        if S <= 0:
            return 1e12
        theta = n / S
        return -(n * np.log(theta) - n * log_sig - (theta + 1.0) * S)

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(negprof, bounds=(-10, 25), method='bounded',
                          options={'xatol': 1e-10})
    sigma = float(np.exp(res.x))
    S = np.sum(np.log1p(w / sigma))
    theta = n / S
    ll = n * np.log(theta) - n * np.log(sigma) - (theta + 1.0) * S
    return theta, sigma, ll


# ── Score tests ───────────────────────────────────────────────────────────────

def lmp_test(x, mu=None):
    """Goerlich (2013) LM test of H0: Pareto I within Pareto II.

    LMP = n (theta+2)(theta+1)^4/theta * [theta/(theta+1) − mean(mu/x)]^2,
    asymptotically chi2(1) under H0. Returns (stat, pval).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if mu is None:
        mu = x.min()
    theta = n / np.sum(np.log(x / mu))
    m = np.mean(mu / x)
    stat = n * ((theta + 2.0) * (theta + 1.0) ** 4 / theta) \
        * (theta / (theta + 1.0) - m) ** 2
    return stat, stats.chi2.sf(stat, 1)


def lmz_test(x, mu=None):
    """Urzúa (2000) LM test of H0: Zipf (theta=1, sigma=mu) within Pareto II.

    LMZ = 4n (z1² + 6 z1 z2 + 12 z2²), chi2(2) under H0. Returns (stat, pval).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if mu is None:
        mu = x.min()
    z1 = 1.0 - np.mean(np.log(x / mu))
    z2 = 0.5 - np.mean(mu / x)
    stat = 4.0 * n * (z1 ** 2 + 6.0 * z1 * z2 + 12.0 * z2 ** 2)
    return stat, stats.chi2.sf(stat, 2)


def pwl_test(x, mu):
    """Urzúa (2020) score test of Pareto I within Pareto IV.

    Requires x > mu strictly. Restricted MLE theta = n/sum(log(x/mu)).
    Score vector d = (d1, d2, 0) w.r.t. (sigma, lambda, theta) at
    sigma=mu, lambda=1:
        d1 = n·theta/mu − (theta+1)·Σ 1/x
        d2 = −n + theta·Σ log((x−mu)/mu) − (theta+1)·Σ (mu/x)·log((x−mu)/mu)
    Information matrix from Brazauskas (2002) with the restrictions imposed.
    PWL = d' I^{-1} d ~ chi2(2) under H0. Returns (stat, pval).
    """
    x = np.asarray(x, dtype=float)
    x = x[x > mu]
    n = x.size
    b = n / np.sum(np.log(x / mu))          # restricted MLE of theta

    lw = np.log((x - mu) / mu)
    d1 = n * b / mu - (b + 1.0) * np.sum(1.0 / x)
    d2 = -n + b * np.sum(lw) - (b + 1.0) * np.sum((mu / x) * lw)
    d = np.array([d1, d2, 0.0])

    p = digamma(b) - digamma(1.0) - 1.0     # p(theta)
    q = polygamma(1, b) + polygamma(1, 1.0)  # q(theta)

    I = np.empty((3, 3))
    I[0, 0] = b / (mu ** 2 * (b + 2.0))
    I[0, 1] = I[1, 0] = -(b * p + 1.0) / (mu * (b + 2.0))
    I[0, 2] = I[2, 0] = -1.0 / (mu * (b + 1.0))
    I[1, 1] = (b * (p ** 2 + q) + 2.0 * (p + 1.0)) / (b + 2.0)
    I[1, 2] = I[2, 1] = p / (b + 1.0)
    I[2, 2] = 1.0 / b ** 2
    I *= n

    stat = float(d @ np.linalg.solve(I, d))
    return stat, stats.chi2.sf(stat, 2)


# ── Finite-sample (parametric bootstrap) p-values ─────────────────────────────

def bootstrap_pvalue(x, test, mu=None, reps=999, rng=None, zipf_null=False):
    """Parametric-bootstrap p-value for a score test under Pareto I H0.

    Simulates `reps` Pareto I samples of the same size (theta at the
    restricted MLE, or theta=1 if zipf_null) and returns the share of
    simulated statistics >= observed. The location convention of `test`
    is preserved: for pwl_test, mu must be supplied and is held fixed;
    otherwise mu is re-set to the simulated minimum, as in the observed data.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = x.size
    mu0 = x.min() if mu is None else mu
    theta0 = 1.0 if zipf_null else n / np.sum(np.log(x[x > 0] / mu0)
                                              if mu is None else np.log(x / mu0))
    if test is pwl_test:
        stat_obs = test(x, mu0)[0]
    else:
        stat_obs = test(x, mu0 if mu is not None else None)[0]
    count = 0
    for _ in range(reps):
        u = rng.random(n)
        xb = mu0 * u ** (-1.0 / theta0)
        if test is pwl_test:
            sb = test(xb, mu0)[0]
        else:
            sb = test(xb, None)[0]
        if sb >= stat_obs:
            count += 1
    return (count + 1) / (reps + 1)


# ── Non-nested comparison: Pareto I vs truncated lognormal ────────────────────

def vuong_pareto_lognormal(x, mu=None):
    """Vuong test of Pareto I against left-truncated lognormal on [mu, inf).

    Positive statistic favours Pareto I; |stat|>1.96 significant at 5%.
    Returns (stat, pval, mu_ln, sig_ln).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if mu is None:
        mu = x.min()
    theta = n / np.sum(np.log(x / mu))
    log_f = np.log(theta) + theta * np.log(mu) - (theta + 1.0) * np.log(x)

    # ML fit of truncated lognormal by direct optimisation
    from scipy.optimize import minimize
    lx = np.log(x)

    def negll(par):
        m, ls = par
        s = np.exp(ls)
        z = (np.log(mu) - m) / s
        return -(np.sum(stats.norm.logpdf(lx, m, s)) - np.sum(lx)
                 - n * np.log(stats.norm.sf(z)))

    res = minimize(negll, x0=[np.mean(lx), np.log(np.std(lx) + 1e-6)],
                   method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
    m, s = res.x[0], float(np.exp(res.x[1]))
    z = (np.log(mu) - m) / s
    log_g = stats.norm.logpdf(lx, m, s) - lx - np.log(stats.norm.sf(z))

    li = log_f - log_g
    denom = np.std(li, ddof=1)
    stat = np.sqrt(n) * np.mean(li) / denom if denom > 0 else 0.0
    return stat, 2.0 * stats.norm.sf(abs(stat)), m, s


# ── Concentration measures ────────────────────────────────────────────────────

def concentration(x):
    """Returns dict with CR1, CR5, CR10, CR50, HHI, Gini for a size vector."""
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    sh = x / x.sum()
    n = x.size
    asc = sh[::-1]
    gini = np.sum((2 * np.arange(1, n + 1) - n - 1) * asc) / (n * asc.sum())
    return {'CR1': sh[:1].sum(), 'CR5': sh[:5].sum(), 'CR10': sh[:10].sum(),
            'CR50': sh[:50].sum(), 'HHI': np.sum(sh ** 2), 'Gini': gini}
