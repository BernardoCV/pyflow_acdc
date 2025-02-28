Installation
===========

Requirements
------------
* Python 3.8+
* Dependencies:
    * numpy
    * pandas
    * networkx
    * matplotlib
    * scipy
    * prettytable
    * plotly
    * folium
    * geopandas
    * scikit-learn-extra
    * scikit-learn

* Optional dependencies:
    * Optimal power flow:  
     * pyomo
     * ipopt
    * Dash:
     * dash


Install from PyPI
----------------
::

    pip install pyflow-acdc


Install from source
------------------
::

    git clone https://github.com/adored-project/pyflow_acdc.git
    cd pyflow_acdc
    pip install -e .



Making Changes
-------------

1. Create a new branch for your changes::

    git checkout -b new-branch-name
    git push origin new-branch-name

2. To push your changes to the remote repository::

    git add .
    git commit -m "Description of your changes"
    git pull origin new-branch-name
    git push origin new-branch-name

3. To pull the latest changes from the remote repository::

    git pull origin main

.. note::
    To merge your changes into the main branch please contact the repository owner.

Additional Dependencies
---------------------

For OPF functionality
^^^^^^^^^^^^^^^^^^^^
Install the following packages::

    pip install pyomo
    pip install ipopt

For Dash Interface
^^^^^^^^^^^^^^^^^
Install the following packages::

    pip install dash