Citing pyflow-acdc
==================

If you use pyflow-acdc in your work, please cite the following paper:

For the general framework:

B. C. Valerio, V. A. Lacerda, M. Cheah-Mane, P. Gebraad and O. Gomis-Bellmunt, "An optimal power flow tool for AC/DC systems, applied to the analysis of the North Sea Grid for offshore wind integration," in IEEE Transactions on Power Systems, doi: 10.1109/TPWRS.2025.3533889.

.. code-block:: bibtex

    @ARTICLE{bcv2025opf,
    author={Valerio, Bernardo Castro and Lacerda, Vinicius A. and Cheah-Mane, Marc and Gebraad, Pieter and Gomis-Bellmunt, Oriol},
    journal={IEEE Transactions on Power Systems},
    title={An optimal power flow tool for AC/DC systems, applied to the analysis of the North Sea Grid for offshore wind integration},
    year={2025},
    volume={},
    number={},
    pages={1-14},
    keywords={Renewable energy sources;Load flow;Voltage;Optimization;Topology;Hybrid power systems;AC-DC power converters;Reactive power;Load modeling;Europe;HVDC;Hybrid AC/DC;Multiterminal DC network;North Sea;Offshore Wind;Optimal Power Flow;Power System modelling},
    doi={10.1109/TPWRS.2025.3533889}}


For the market-based OPF:

.. code-block:: bibtex

    @ARTICLE{bcv2025market,
    author={Valerio, Bernardo Castro and Lacerda, Vinicius A. and Cheah-Mane, Marc and Gebraad, Pieter and Gomis-Bellmunt, Oriol},
    title={Optimizing Offshore Wind Integration through Multi-Terminal DC Grids: A Market-Based OPF Framework for the North Sea Interconnectors},
    year={2025},
    volume={},
    number={},
    pages={1-6},
    keywords={HVDC, INTERCONNECTOR, MTDC, OPTIMAL POWER FLOW, PRICE ZONES}
    }


For transmission expansion planning:

.. code-block:: bibtex

    @article{VALERIO2026111459,
    title = {Transmission expansion planning for hybrid AC/DC grids using a mixed-integer non-linear programming approach},
    journal = {International Journal of Electrical Power & Energy Systems},
    volume = {174},
    pages = {111459},
    year = {2026},
    issn = {0142-0615},
    doi = {https://doi.org/10.1016/j.ijepes.2025.111459},
    url = {https://www.sciencedirect.com/science/article/pii/S0142061525010075},
    author = {Bernardo Castro Valerio and Marc Cheah-Mane and Vinicius A. Lacerda and Pieter Gebraad and Oriol Gomis-Bellmunt},
    keywords = {Transmission expansion, Optimal power flow, Non-linear, Hybrid AC/DC}}


For array optimization:

Castro Valerio, B., Gebraad, P. M. O., Cheah-Mane, M., A. Lacerda, V., and Gomis-Bellmunt, O.: A multi-stage methodology for wind park inter-array cabling: graph preparation, layout, and sizing, Wind Energ. Sci. Discuss. [preprint], https://doi.org/10.5194/wes-2026-53, in review, 2026.

For BESS / hydrogen / energy-island operation:

If you use :class:`~pyflow_acdc.Storage_AC`, :class:`~pyflow_acdc.Storage_DC`,
:func:`~pyflow_acdc.add_storage`, :class:`~pyflow_acdc.Electrolyzer`,
:func:`~pyflow_acdc.add_electrolyzer`, or
:func:`~pyflow_acdc.window_nl_opf`, please also cite:

M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt, *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026.

User guides: :doc:`usage_storage` (BESS, §3.3), :doc:`usage_hydrogen` (electrolyzer, §3.4).

.. code-block:: bibtex

    @Article{wes-11-349-2026,
      author  = {Useche-Arteaga, M. and Gebraad, P. and Lacerda, V. and Cheah-Mane, M. and Gomis-Bellmunt, O.},
      title   = {Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the {Princess Elisabeth Energy Island}},
      journal = {Wind Energy Science},
      volume  = {11},
      year    = {2026},
      number  = {2},
      pages   = {349--372},
      url     = {https://wes.copernicus.org/articles/11/349/2026/},
      doi     = {10.5194/wes-11-349-2026}
    }
