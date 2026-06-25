# Case118 benchmark time-series data

CSV files for ``pyf.cases["case118_TEP_benchmark"]()``:

| File | Role |
|------|------|
| `118_benchmark_loads.csv` | Per-bus load factors (`Load`); row 0 = bus name |
| `118_benchmark_rgen.csv` | Renewable availability per generator; zone series use the first gen in each `Wind_*` / `Solar_1` zone |
| `118_benchmark_wl.csv` | Legacy combined wind + zonal-load file (superseded by loads + rgen) |
