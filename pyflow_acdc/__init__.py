# -*- coding: utf-8 -*-
"""
PyFlow-ACDC initialization module.
Provides grid simulation and power flow analysis functionality.
"""
from pathlib import Path
import importlib.util

from .Class_editor import *
from .Export_files import *
from .Time_series import *
from .ACDC_PF import *
from .ACDC_OPF import *
from .Graph_and_plot import *
from .Market_Coeff import *

# Define what should be available when users do: from pyflow_acdc import *
__all__ = [
    # Add Grid Elements
    'add_AC_node',
    'add_DC_node',
    'add_line_AC',
    'add_line_DC',
    'add_ACDC_converter',
    'add_gen',
    'add_extGrid',
    'add_RenSource',
    'add_generators_fromcsv',
    
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
    'reset_all_class',
    
    # Power Flow
    'AC_PowerFlow',
    'DC_PowerFlow',
    'ACDC_sequential',
    
    # OPF
    'OPF_ACDC',
    'OPF_solve',
    'OPF_obj',
    'OPF_line_res',
    'OPF_price_priceZone',
    'OPF_conv_results',
    'Translate_pyf_OPF',
    
    # Time Series Analysis
    'Time_series_PF',
    'TS_ACDC_PF',
    'TS_ACDC_OPF_parallel',
    'Time_series_statistics',
    'results_TS_OPF',
    
    # Export
    'export_results_to_excel',
    'export_OPF_results_to_excel',
    
    # Visualization
    'plot_network',
    'plot_Graph',
    'plot_neighbour_graph',
    'plot_time_series',
    'plot_TS_res',
    'plot_statistics',
    'plot_folium',
    'plot_map',
    
    # Market Analysis
    'calculate_market_coefficients'
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