"""Docs: api\plotting.rst — Full grid visualization as a network graph"""
import pyflow_acdc as pyf
grid,res = pyf.cases['case24_3zones_acdc']()
pyf.plot_graph(grid)

