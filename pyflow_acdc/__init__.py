# -*- coding: utf-8 -*-
"""
PyFlow-ACDC initialization module.
Provides grid simulation and power flow analysis functionality.
"""
from pathlib import Path
import importlib.util

from .Results_class import *
from .Class_editor import *
from .Grid_creator import *
from .Classes import *

from .Export_files import *
from .Time_series import *

from .ACDC_PF import *

from .Graph_and_plot import *
from .Market_Coeff import *


# Try to import OPF module if pyomo is available
try:
    from .ACDC_OPF import *
    from .ACDC_TEP import *
    
    HAS_OPF = True
except ImportError:
    HAS_OPF = False
    

try:
    from .Graph_Dash import *
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    

try:
    from .Mapping import *
    HAS_MAPPING = True
except ImportError:
    HAS_MAPPING = False

try:
    from .Time_series_clustering import *
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    
# Define what should be available when users do: from pyflow_acdc import *
__all__ = [
    # Results
    'Results',
    # Grid
    'Grid', 
    'Node_AC',
    'Node_DC',
    'Line_AC',
    'Line_DC',
    'AC_DC_converter',

    # Add Grid Elements
    'add_AC_node',
    'add_DC_node',
    'add_line_AC',
    'add_line_DC',
    'add_ACDC_converter',
    'add_DCDC_converter',
    'add_gen',
    'add_gen_DC',
    'add_extGrid',
    'add_RenSource',
    'add_generators',
    
    # Add Zones
    'add_RenSource_zone',
    'add_price_zone',
    'add_MTDC_price_zone',
    'add_offshore_price_zone',
    
    # Add Time Series
    'add_TimeSeries',
    
    # Grid Creation and Import
    'Create_grid_from_data',
    'Create_grid_from_mat',
    'Extend_grid_from_data',
    
    # Line Modifications
    'change_line_AC_to_expandable',
    'change_line_AC_to_tap_transformer',
    
    # Zone Assignments
    'assign_RenToZone',
    'assign_nodeToPrice_Zone',
    'assign_ConvToPrice_Zone',
    
    # Parameter Calculations
    'Cable_parameters',
    'Converter_parameters',
    
    # Utility Functions
    'pol2cart',
    'cart2pol',
    'pol2cartz',
    'cartz2pol',
    'initialize_pyflowacdc',
    
    # Power Flow
    'AC_PowerFlow',
    'DC_PowerFlow',
    'ACDC_sequential',
    'Power_flow',
    
    # OPF
    'Optimal_PF',
    'OPF_solve',
    'OPF_obj',
    'OPF_line_res',
    'OPF_price_priceZone',
    'OPF_conv_results',
    'Translate_pyf_OPF',
    
    # TEP
    'transmission_expansion',
    'multi_scenario_TEP',
    'update_grid_price_zone_data',
    'expand_elements_from_pd',
    'repurpose_element_from_pd',
    'update_attributes',
    'Expand_element',
    'Translate_pd_TEP',
    'export_TEP_TS_results_to_excel',

    # Time Series Analysis
    'Time_series_PF',
    'TS_ACDC_PF',
    'TS_ACDC_OPF',
    'TS_ACDC_OPF_parallel',
    'Time_series_statistics',
    'results_TS_OPF',
    'cluster_TS',
    'run_clustering_analysis_and_plot',
    'identify_correlations',
    'update_grid_data',
    
    # Export
    'export_results_to_excel',
    'export_OPF_results_to_excel',
    'save_grid_to_file',
    'save_grid_to_matlab',
    # Visualization
    
    'plot_Graph',
    'Time_series_prob',
    'plot_neighbour_graph',
    'plot_TS_res',
    'save_network_svg',
    
    # Market Analysis
    'price_zone_data_pd',
    'price_zone_coef_data',
    'plot_curves',
    'clean_entsoe_data',

    'run_dash',
]

# Dynamically load all .py files in the 'cases/' folder
case_folder = Path(__file__).parent / "example_grids"

# Namespace for all loaded cases
cases = {}

# Load each .py file in the cases folder
for case_file in case_folder.glob("*.py"):
    module_name = case_file.stem  # Get the file name without extension
    spec = importlib.util.spec_from_file_location(module_name, case_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Add all public functions from the module to the `cases` namespace
    cases.update({name: obj for name, obj in vars(module).items() if not name.startswith("_")})

# Optional: Add all cases to this module's global namespace
globals().update(cases)    

