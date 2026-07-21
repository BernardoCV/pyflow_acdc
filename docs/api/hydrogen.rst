Green hydrogen / electrolyzer
=============================

:class:`~pyflow_acdc.Electrolyzer` and :func:`~pyflow_acdc.add_electrolyzer`
attach operation-only electrolyzers to AC or DC buses. Active power is a
**load**; inventory ``mass_H2`` is tracked in **kg**. Nonlinear OPF and
coupled multi-hour H₂ inventory use :func:`~pyflow_acdc.window_nl_opf`.

User guide: :doc:`../usage_hydrogen`.

Electrolyzer class
------------------

.. autoclass:: pyflow_acdc.Electrolyzer
   :members:
   :exclude-members: reset_class

Add electrolyzer
----------------

.. autofunction:: pyflow_acdc.add_electrolyzer

Modelling note
--------------

When ``grid.H2`` is true, :mod:`~pyflow_acdc.ACDC_OPF_NL_model` adds
``P_electrolyzer``, optional AC ``Q_electrolyzer``, and ``mass_H2`` with a
one-step inventory balance for snapshot OPF. Multi-hour coupled runs use
:func:`~pyflow_acdc.window_nl_opf` (parent ``window_h2_constraints``; terminal
``H2_mass_final`` when set).

Production each frame (Useche-Arteaga et al. 2026 §3.4; ``c_h`` every hour):

.. math::

   h = b_h\, P_e\, S_{\mathrm{base}}\, \Delta t + c_h

   M_t = M_{t-1} + h

The formulation follows Useche-Arteaga et al. (2026) [#useche2026]_.
See :doc:`../citing` for the BibTeX entry.

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
