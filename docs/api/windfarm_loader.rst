Wind-Farm Case Loader
=====================

Loads bundled wind-farm case grids (pickle) together with optional GeoJSON
geographic context (development area, exclusions, export cables).

Requires ``pyflow_acdc[mapping]`` (``shapely``) for GeoJSON parsing.

Functions are found in ``pyflow_acdc.windfarm_loader``.

.. autofunction:: pyflow_acdc.windfarm_loader.load_case_grid_and_geo

