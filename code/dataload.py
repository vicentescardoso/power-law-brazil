"""Load the 1999-2024 Valor top-1000 fiscal-year panel.

The panel is built from three source workbooks:

* ``painel_valor_1999_2024.xlsx`` -- the derived annual ranking workbook;
* ``historico_valor_2001_2025.xlsx`` -- the independent historical rankings,
  used to rebuild fiscal 2012-2014;
* ``valor_2020_edition.xlsx`` -- the original 2020 annual edition, used to
  recover the fiscal-2019 rank-779 observation that is shifted in the
  derived workbook.

The loader applies two documented repairs:

R1. Fiscal 2012-2014 are taken from the historical workbook (sheets
    ``2013``, ``2014`` and ``2015``), because the corresponding sheets in the
    derived workbook carry fiscal-2014 values (2013 and 2014) or the same
    values rotated by one rank (2015).

R2. The fiscal-2019 rank-779 row of the derived workbook is shifted one cell
    to the left. The original company (SG Comercio Exterior), state (MG) and
    revenue (R$ 657.0 million) are recovered from the 2020 annual edition.

The cleaned panel is written by :mod:`run` as an output; it is never edited by
hand.
"""
from __future__ import annotations

import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PAINEL_XLSX = BASE / "data" / "raw" / "painel_valor_1999_2024.xlsx"
HIST_XLSX = BASE / "data" / "raw" / "historico_valor_2001_2025.xlsx"
ANNUAL_2020_XLSX = BASE / "data" / "raw" / "valor_2020_edition.xlsx"

UF = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO", "EX",
}

# Fiscal years rebuilt from the historical workbook (repair R1).
SPLICED_YEARS = {2012: "2013", 2013: "2014", 2014: "2015"}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if unicodedata.category(c) != "Mn"
    )


def clean_name(name) -> str:
    """Normalise company names for cross-year matching."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = _strip_accents(str(name)).lower()
    s = re.sub(r"[´`'\"!+|]", " ", s)
    s = re.sub(r"[\*\.\,\-\/&\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(3):
        s = re.sub(r"(\s+\d{1,2})+$", "", s).strip()
        m = re.match(r"^(.*[a-z]{2,})\d{1,2}$", s)
        if m:
            s = m.group(1).strip()
        else:
            break
    for suf in [
        r"\bs\s*a\b", r"\bsa\b", r"\bltda\b", r"\bcia\b", r"\bcompanhia\b",
        r"\bgrupo\b", r"\bparticipacoes\b", r"\bpart\b", r"\bholding\b",
        r"\bholdings\b", r"\bs\b", r"\ba\b",
    ]:
        s = re.sub(suf, "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


# Manually vetted aliases used to link trade names across editions.
ALIASES = {
    "rio de janeiro aeroporto": "rio de janeiro aeroportos",
    "distribuidora santalucia": "distribuidora santa lucia",
    "beneficiencia portuguesa": "beneficencia portuguesa",
    "stefanini it solutions": "stefanini it solution",
    "ouro fino saude animal": "ourofino saude animal",
    "supermercado mundial": "supermercados mundial",
    "supermercados bahamas": "supermercado bahamas",
    "supermercado avenida": "supermercados avenida",
    "ourofino agrociencia": "ouro fino agrociencia",
    "milenia agrociencias": "milenia agro ciencias",
    "farmacias pague menos": "farmacia pague menos",
    "cerradinho bioenergia": "cerradinho bionergia",
    "rodovia das colinas": "rodovias das colinas",
    "china brasil tabaco": "china brasil tabacos",
    "latincinios jussara": "laticinios jussara",
    "hisparmar satelites": "hispamar satelites",
    "bevape bioenergia": "bevap bioenergia",
    "louis dreyfus c agroindustrial": "louis dreyfus agroindustrial",
    "petronas lubrificantes brasil": "petronas lubricantes brasil",
    "timbo trading": "timbro trading",
    "tenaris confab": "tenarisconfab",
    "usina estiva": "usina estivas",
    "destilaria santa isabel": "destilaria santa izabel",
    "bm f bovespa": "bm fbovespa",
    "accorhotels": "accor hotels",
    "supermercado barbosa": "supermercado g barbosa",
    "cantustore": "cantu store",
    "sol coqueria tubarao": "sol coqueira tubarao",
    "expocacer": "expocaccer",
    "energetica suape il": "energetica suape ii",
    "universidade de nove de julho": "universidade nove de julho",
    "state grid brasil": "state grid brazil",
    "smart fit": "smartfit",
    "fujifilm": "fuji film",
    "diebold procomp amazonia": "diebold procomp da amazonia",
    "suzano de papel e celulose": "suzano papel e celulose",
    "zaffari e bourbon": "zaffari bourbon",
    "usina sao jose estiva": "usina sao jose da estiva",
    "owens illinois do brasil": "owens illinois brasil",
    "embranco": "embraco",
    "dana industrias": "dana industrial",
    "clariont": "clarion",
    "refinaria manguinhos": "refinaria de manguinhos",
    "mercado joanim": "mercado joanin",
    "comere energia": "comerc energia",
    "usina nandini": "usina nardini",
    "mgb sup mambo e giga atacado": "mgb mambo e giga atacado",
    "lojas yamada": "lojas y yamada",
    "ajinomoto brasil": "ajinomoto do brasil",
    "randstad do brasil": "randstad brasil",
    "brinks": "brink",
    "santa casa de sao paulo": "santa casa de paulo",
    "muller bebidas": "muller de bebidas",
    "transwoiff": "transwolff",
    "polo filme": "polo films",
    # The 1999-2000 FGV block uses full legal names; Valor 2001+ uses trade names.
    "petroleo brasileiro petrobras": "petrobras",
}


def _parse_painel_sheet(xls, sheet: str) -> pd.DataFrame:
    """Parse one year sheet of the derived workbook."""
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    hdr_row = None
    for i in range(min(60, len(df))):
        if (df.iloc[i].astype(str).str.strip() == "Empresa").any():
            hdr_row = i
            break
    if hdr_row is None:
        raise ValueError(f"sheet {sheet}: 'Empresa' header not found")
    header_zone = df.iloc[max(0, hdr_row - 2):hdr_row + 5]

    rev_col = name_col = None
    for c in df.columns:
        vals = header_zone[c].astype(str).str.strip()
        if rev_col is None and (vals.isin(["Sales", "Receita"])).any():
            rev_col = c
        if name_col is None and (vals == "Empresa").any():
            name_col = c
    if rev_col is None:
        raise ValueError(f"sheet {sheet}: revenue header not found")
    if name_col is not None and df.iloc[hdr_row + 1:][name_col].notna().sum() == 0:
        for c in df.columns:
            if header_zone[c].astype(str).str.strip().eq("Razão Social").any():
                name_col = c
                break

    rows = []
    for i in range(hdr_row + 1, len(df)):
        comp = df.iloc[i, name_col]
        rev = pd.to_numeric(df.iloc[i, rev_col], errors="coerce")
        if not (isinstance(comp, str) and comp.strip() and pd.notna(rev) and rev > 0):
            continue
        comp = comp.strip().rstrip("*").strip()
        if comp.upper() in UF:  # repair R2: stray state-code cells
            continue
        rows.append((comp, float(rev)))
    out = pd.DataFrame(rows, columns=["company", "sales"])
    if out["sales"].max() > 1e9:  # older sheets store plain R$
        out["sales"] /= 1e6
    return out.sort_values("sales", ascending=False).reset_index(drop=True)


def _parse_historico_sheet(xls, sheet: str) -> pd.DataFrame:
    """Parse one edition sheet of the historical workbook.

    Sheet ``YYYY`` reports fiscal ``YYYY-1``.
    """
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    hdr_row = None
    for i in range(min(25, len(df))):
        if df.iloc[i].astype(str).str.strip().eq("Classificação").any():
            hdr_row = i
            break
    if hdr_row is None:
        raise ValueError(f"sheet {sheet}: 'Classificacao' header not found")
    emp_col = rec_col = None
    for c in range(df.shape[1]):
        for rr in range(max(0, hdr_row - 2), min(len(df), hdr_row + 5)):
            v = str(df.iloc[rr, c]).strip()
            if emp_col is None and v == "Empresa":
                emp_col = c
            if rec_col is None and v.startswith("Receita"):
                rec_col = c
    if emp_col is None or rec_col is None:
        raise ValueError(f"sheet {sheet}: Empresa/Receita columns not found")

    rows = []
    for i in range(hdr_row + 1, len(df)):
        comp = df.iloc[i, emp_col]
        if isinstance(comp, str) and comp.strip() == "Empresa":
            continue  # repeated page headers
        rev = pd.to_numeric(df.iloc[i, rec_col], errors="coerce")
        if not (isinstance(comp, str) and comp.strip() and pd.notna(rev) and rev > 0):
            continue
        comp = comp.strip().rstrip("*").strip()
        if comp.upper() in UF:
            continue
        rows.append((comp, float(rev)))
    out = pd.DataFrame(rows, columns=["company", "sales"])
    if out["sales"].max() > 1e7:  # historical workbook stores plain R$
        out["sales"] /= 1e6
    return out.sort_values("sales", ascending=False).reset_index(drop=True)


def load_panel(verbose: bool = False, restore_2019: bool = True) -> dict[int, pd.DataFrame]:
    """Return ``{fiscal_year: DataFrame(company, clean, sales)}``.

    Parameters
    ----------
    verbose:
        Print one line per fiscal year.
    restore_2019:
        If true (default), append the recovered fiscal-2019 rank-779 firm.
        If false, return the 999-firm fiscal-2019 sample used by the
        999-replication historical bootstrap calibration.
    """
    panel: dict[int, pd.DataFrame] = {}
    xls = pd.ExcelFile(PAINEL_XLSX)
    hist_xls = None
    for fy in range(1999, 2025):
        sheet = str(fy + 1)  # edition year = fiscal year + 1
        d = _parse_painel_sheet(xls, sheet)

        if fy in SPLICED_YEARS:  # repair R1
            if hist_xls is None:
                hist_xls = pd.ExcelFile(HIST_XLSX)
            d_fix = _parse_historico_sheet(hist_xls, SPLICED_YEARS[fy])
            if verbose:
                print(
                    f"[splice] fiscal {fy}: replaced derived sheet '{sheet}' "
                    f"with historical sheet '{SPLICED_YEARS[fy]}' "
                    f"(n={len(d_fix)}, top={d_fix.iloc[0]['company']})"
                )
            d = d_fix

        if fy == 2019 and restore_2019:
            raw = pd.read_excel(ANNUAL_2020_XLSX, header=None)
            match = raw.loc[pd.to_numeric(raw[0], errors="coerce").eq(779)]
            if len(match) != 1:
                raise ValueError("2019 source rank 779 not unique")
            source = match.iloc[0]
            if not (
                str(source[1]).strip() == "SG Comércio Exterior"
                and str(source[2]).strip() == "MG"
                and abs(float(source[4]) - 657.0) < 1e-10
            ):
                raise ValueError("2019 source rank 779 no longer matches the audited repair")
            d = pd.concat(
                [
                    d,
                    pd.DataFrame(
                        [{"company": str(source[1]).strip(), "sales": float(source[4])}]
                    ),
                ],
                ignore_index=True,
            )

        panel[fy] = d.sort_values("sales", ascending=False).reset_index(drop=True)

    for fy in panel:
        panel[fy]["clean"] = panel[fy]["company"].map(clean_name)

    if verbose:
        for fy in sorted(panel):
            top = panel[fy].iloc[0]
            print(
                f"fiscal {fy}: n={len(panel[fy]):4d}  "
                f"top={top['company'][:25]:25s} {top['sales']:>12,.0f} R$ mn"
            )
    return panel


def without_2019_restoration(panel: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Return a copy of ``panel`` with the recovered 2019 rank-779 firm removed.

    The result is the 999-firm fiscal-2019 sample used by the historical
    999-replication bootstrap calibration.
    """
    out = {fy: df.copy() for fy, df in panel.items()}
    mask = ~(
        (out[2019]["company"] == "SG Comércio Exterior")
        & (out[2019]["sales"] == 657.0)
    )
    out[2019] = out[2019].loc[mask].reset_index(drop=True)
    return out


def qa_panel(panel: dict[int, pd.DataFrame], verbose: bool = True) -> list[str]:
    """Run automated data-quality checks; return a list of issues."""
    issues: list[str] = []
    years = sorted(panel)
    for fy in years:
        names = panel[fy]["company"].str.upper().str.strip()
        bad = names[names.isin(UF)]
        if len(bad):
            issues.append(f"fiscal {fy}: state-code company cells {bad.tolist()}")
        expected = 999 if (fy == 2019 and len(panel[fy]) == 999) else 1000
        if len(panel[fy]) != expected:
            issues.append(
                f"fiscal {fy}: expected {expected} firms, found {len(panel[fy])}"
            )

    # Adjacent-year shift detector: s_t[k] ~= s_{t-1}[k+1] for the top ranks.
    for a, b in zip(years[:-1], years[1:]):
        sa = panel[a]["sales"].values
        sb = panel[b]["sales"].values
        n = min(200, len(sa) - 1, len(sb))
        if n < 50:
            continue
        if np.allclose(sb[:n], sa[1:n + 1]):
            issues.append(f"years {a}->{b}: sales vector shifted by one rank")
        ratio = sb[0] / sa[0]
        if not (0.5 < ratio < 2.5):
            issues.append(
                f"years {a}->{b}: top-firm revenue ratio {ratio:.2f} outside [0.5, 2.5]"
            )
    if verbose:
        if issues:
            for msg in issues:
                print("[QA] ISSUE:", msg)
        else:
            print("[QA] all checks passed")
    return issues


if __name__ == "__main__":
    load_panel(verbose=True)
