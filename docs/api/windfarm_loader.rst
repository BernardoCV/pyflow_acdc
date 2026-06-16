Wind-Farm Case Loader
=====================

Loads bundled wind-farm case grids (pickle) together with optional GeoJSON
geographic context (development area, exclusions, export cables).

Requires ``pyflow_acdc[mapping]`` (``shapely``) for GeoJSON parsing.

Functions are found in ``pyflow_acdc.windfarm_loader``.

.. autofunction:: pyflow_acdc.windfarm_loader.load_case_grid_and_geo

   **Example**

   .. code-block:: python

       import pyflow_acdc as pyf

       grid, res = pyf.windfarm_loader.load_case_grid_and_geo("MorayEast")

       # Geometry context is attached to grid for folium / array plotting:
       # grid.dev_polygon, grid.export_cables, grid.exclusion_zones, ...
