Battery energy storage
======================

:class:`~pyflow_acdc.Storage` and :func:`~pyflow_acdc.add_storage` attach
operation-only BESS elements to AC or DC buses. Nonlinear OPF constraints and
coupled multi-hour runs use :func:`~pyflow_acdc.window_nl_opf`.
Long series: :func:`~pyflow_acdc.rolling_window_nl_opf`.
Myopic sequential hours: :func:`~pyflow_acdc.ts_acdc_opf` with optional
``ObjRule['SoC_deviation']`` (soft ``soc_ref``).

User guide: :doc:`../usage_storage`.

Storage class
-------------

.. autoclass:: pyflow_acdc.Storage
   :members:
   :exclude-members: reset_class

Add storage
-----------

.. autofunction:: pyflow_acdc.add_storage

.. autofunction:: pyflow_acdc.window_nl_opf

.. autofunction:: pyflow_acdc.rolling_window_nl_opf

Modelling note
--------------

The BESS constraints (SoC dynamics, AC S-circle, DC net-P limit) are
implemented in :mod:`~pyflow_acdc.ACDC_OPF_NL_model` for snapshot OPF when
``grid.ESS`` is true. Multi-hour coupled runs use :func:`~pyflow_acdc.window_nl_opf`.
The formulation follows Useche-Arteaga et al. (2026)
[#useche2026]_. See :doc:`../citing` for the BibTeX entry.

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
