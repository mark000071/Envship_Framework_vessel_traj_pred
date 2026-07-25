# Result tables

`aggregate_multiseed.py` reads the per-seed result JSONs written by the training scripts
(under `$ENVSHIP_RES_DIR`, default `results/`) and writes `summary.json` with the 5-seed
mean and standard deviation of ADE / FDE for each model, for all three main tables.
`make_tables.py` turns that summary into the LaTeX table bodies.

```
python tables/aggregate_multiseed.py
python tables/make_tables.py
```
