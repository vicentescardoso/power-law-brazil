# Data

## Source

The panel combines two Brazilian firm-size series:

* fiscal years 1999–2000: the IBRE-FGV series of the largest firms, published
  in *Conjuntura Econômica*;
* fiscal years 2001–2024: the annual ranking of the 1,000 largest companies by
  net revenue published by *Valor Econômico* (the "Valor 1000" yearbook).

The paper uses net revenue in R$ million and the annual ranking of the 1,000
largest firms, so every cross-section has exactly 1,000 observations.

## Files

| File | Description |
|---|---|
| `raw/painel_valor_1999_2024.xlsx` | derived annual ranking workbook: one sheet per edition, plus a `base` sheet with the pooled 1999–2019 extract and a `parametros` sheet with auxiliary calculations |
| `raw/historico_valor_2001_2025.xlsx` | independent compilation of the historical rankings, one sheet per edition plus a `consolidado` sheet; used to rebuild fiscal 2012–2014 |
| `raw/valor_2020_edition.xlsx` | the original 2020 annual edition, used to recover the fiscal-2019 rank-779 observation |

**Sheet naming.** In both ranking workbooks the sheets are named by *edition*
year: sheet `YYYY` reports fiscal year `YYYY − 1`. `code/dataload.py` performs
this mapping, so fiscal years 1999–2024 are read from sheets 2000–2025. The
`year` column inside some sheets is a stale helper column and is not used.

## Repairs

Two data problems are corrected in code, never by editing the workbooks.

**R1 — fiscal 2012–2014.** The corresponding sheets of the derived workbook
carry duplicated or rotated values: sheet 2013 and sheet 2014 both carry
fiscal-2014 values, and sheet 2015 carries the same values rotated by one
rank. These three fiscal years are therefore rebuilt from the independent
historical workbook (sheets 2013, 2014 and 2015).

**R2 — fiscal 2019, rank 779.** In the derived workbook the row for rank 779
is shifted one cell to the left: the company name is lost, the state code
appears in the name column and the revenue cell holds a variance figure. The
original observation (SG Comércio Exterior, MG, R$ 657.0 million) is recovered
from the 2020 annual edition and appended to the fiscal-2019 cross-section.
`code/run.py` writes the repaired panel to `output/cleaned_panel.csv`; the
loader re-checks the source row on every run and fails if it no longer matches.

The 999-firm fiscal-2019 sample used by the historical bootstrap calibration
of Tables 1–3 is obtained by dropping the recovered observation
(`without_2019_restoration` in `code/dataload.py`).

## Checksums

`output/manifest.json` records the SHA-256 digest of each workbook. The values
in the published replication are:

```
303a9e3e21a105c1ee65ef00fe7cf38a90b505ff9589f555f25fc977c6be211c  painel_valor_1999_2024.xlsx
cd9b82bcab51cab125fc5c088f5532224e7806f49fb94ffdaea4ee481412e24b  historico_valor_2001_2025.xlsx
f31140e19c1c111eac4377ff33fb0d692b737ffb90906f08dca531e7e34d9f35  valor_2020_edition.xlsx
```

`code/verify.py` recomputes them, so any accidental edit to a source file is
detected.

## Reuse

The workbooks compile rankings published by *Valor Econômico* and IBRE-FGV.
They are included here so that the published estimates can be reproduced, and
remain subject to the original publishers' rights.
