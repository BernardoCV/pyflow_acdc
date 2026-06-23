import pandas as pd
import pytest

import pyflow_acdc as pyf
from pyflow_acdc.Graph_Dash import create_dash_app
from pyflow_tests._test_solver_deps import (
    dash_missing_for_run_test,
    pyomo_missing_for_run_test,
    require_dash,
    require_pyomo,
)
from pyflow_tests.test_constants import NS_MTDC_MARKET_PRICES_URL, NS_MTDC_WIND_LOAD_URL


def ts_dash():
    grid, results = pyf.cases['NS_MTDC']()

    start = 5990
    end = 6000
    obj = {'Energy_cost': 1}

    TS_MK = pd.read_csv(NS_MTDC_MARKET_PRICES_URL)
    pyf.add_TimeSeries(grid, TS_MK)

    TS_wl = pd.read_csv(NS_MTDC_WIND_LOAD_URL)
    pyf.add_TimeSeries(grid, TS_wl)

    pyf.ts_acdc_opf(grid, start, end, ObjRule=obj)

    app = create_dash_app(grid)
    assert app.layout is not None
    assert app.callback_map

    print('Time series OPF completed and Dash app created')


@pytest.mark.slow
@pytest.mark.integration
def test_ts_dash():
    require_pyomo()
    require_dash()
    ts_dash()


def run_test():
    """Test time series dash functionality."""
    if dash_missing_for_run_test():
        return
    if pyomo_missing_for_run_test():
        return
    ts_dash()


if __name__ == "__main__":
    run_test()
