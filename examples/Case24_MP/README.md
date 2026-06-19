IEEE RTS-96 case24 multi-period TEP investment CSVs for `pyf.cases["case24_MP"]()`.

Files:

- `case24_MP_TEP_inv_series_10.csv` — investment-period load/curvature series
- `case24_MP_TEP_gen_mix_limits.csv` — generation mix limits per period
- `case24_MP_TEP_gen_tracking_data.csv` — generator tracking metadata

Usage:

```python
import importlib.util
from pathlib import Path

import pyflow_acdc as pyf

grid, res = pyf.cases["case24_MP"]()

case_path = Path(pyf.__file__).resolve().parent / "example_grids" / "TEP" / "case24_MP.py"
spec = importlib.util.spec_from_file_location("case24_MP", case_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pyf.add_inv_series(grid, mod._resolve_example_path("case24_MP_TEP_inv_series_10.csv"))
pyf.add_gen_mix_limits(grid, mod._resolve_example_path("case24_MP_TEP_gen_mix_limits.csv"))
```

Raw GitHub base path: `https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/Case24_MP/`
