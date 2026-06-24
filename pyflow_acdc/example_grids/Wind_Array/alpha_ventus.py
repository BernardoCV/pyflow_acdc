from pyflow_acdc.windfarm_loader import load_case_grid_and_geo


def alpha_ventus(cab_types_allowed=3, ns=None):
    grid, res = load_case_grid_and_geo("alpha_ventus")
    grid.cab_types_allowed = cab_types_allowed
    if ns is not None:
        for node in grid.nodes_AC:
            if node.type == "Slack":
                node.ct_limit = ns
    return grid, res