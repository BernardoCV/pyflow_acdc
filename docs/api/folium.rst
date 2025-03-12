Mapping
=======

For this module, you need to have the optional dependendency pyflow_acdc[mapping] installed.


.. py:function:: plot_folium(grid, text='inPu', name='grid_map', tiles="CartoDB Positron", polygon=None, ant_path='None', clustering=True, coloring=None)

   Creates an interactive map visualization using Folium.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid to visualize
        - Required
        - -
      * - ``text``
        - str
        - Hover text format ('inPu' or 'abs')
        - 'inPu'
        - -
      * - ``name``
        - str
        - Output file name
        - 'grid_map'
        - -
      * - ``tiles``
        - str
        - Map style
        - "CartoDB Positron"
        - -
      * - ``ant_path``
        - str
        - Animated paths
        - 'None'
        - -
      * - ``clustering``
        - bool
        - Enable marker clustering
        - True
        - -

   **Features**:

   - Interactive map with zoom/pan
   - Voltage level filtering
   - Component type layers:
     - MVAC Lines (<110kV)
     - HVAC Lines (<300kV)
     - HVAC Lines (<500kV)
     - HVDC Lines
     - Converters
     - Transformers
   - Marker clustering for generators/loads
   - Hover information for components
   - Optional animated power flows

   **Example**

   .. code-block:: python

       grid.plot_folium(name='my_map', text='inPu')

