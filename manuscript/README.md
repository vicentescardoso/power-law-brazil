# Manuscript

* `Power law 20-10 v5.docx` — final manuscript (Word source).
* `Power law 20-10 v5.pdf` — 28-page PDF rendered from the Word file.

`tables/table_1.csv` … `table_6.csv` hold the values exactly as printed in the
manuscript (three decimals, `<0.001` for values below the reporting threshold).
They are the verification targets used by `code/verify.py`; the files written
by `code/run.py` to `output/tables/` carry full precision and a few additional
columns (sample sizes, standard errors, cutoffs).

To regenerate the PDF from the Word file:

```bash
libreoffice --headless --convert-to pdf --outdir manuscript "manuscript/Power law 20-10 v5.docx"
```

(or `flatpak run org.libreoffice.LibreOffice --headless --convert-to pdf …` for
a Flatpak installation).
