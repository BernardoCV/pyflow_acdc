"""Docs: api\\plotting.rst — 3D Grid Plot"""
import pyflow_acdc as pyf
from pyflow_tests.test_constants import CABLE_TYPES_OFF66, MORAY_EAST_CABLE_DECISIONS

grid, res = pyf.cases["moray_east"]()
grid.cab_types_allowed = len(CABLE_TYPES_OFF66)
cable_index = {name: idx for idx, name in enumerate(CABLE_TYPES_OFF66)}
for line in grid.lines_AC_ct:
    line.cable_types = list(CABLE_TYPES_OFF66)
    line.active_config = cable_index.get(MORAY_EAST_CABLE_DECISIONS.get(str(line.name), ""), -1)

pyf.plot_3D(grid, show=True)
