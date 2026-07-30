Green hydrogen / electrolyser
=============================

:class:`~pyflow_acdc.Electrolyser` and :func:`~pyflow_acdc.add_electrolyser`
attach operation-only electrolysers to AC or DC buses. Active power is a
**load**; inventory ``mass_H2`` is tracked in **kg** for snapshot, myopic TS,
and coupled horizon OPF. Nonlinear multi-hour inventory uses
:func:`~pyflow_acdc.window_nl_opf`. Out-of-opt tank resets use
``empty_tank_cycle`` (see :doc:`../usage_hydrogen`).

User guide: :doc:`../usage_hydrogen`.

Electrolyser class
------------------

.. autoclass:: pyflow_acdc.Electrolyser
   :members:
   :exclude-members: reset_class

Add electrolyser
----------------

.. autofunction:: pyflow_acdc.add_electrolyser

Modelling note
--------------

When ``grid.H2`` is true, :mod:`~pyflow_acdc.ACDC_OPF_NL_model` adds
``P_electrolyser``, optional AC ``Q_electrolyser``, and ``mass_H2`` with a
one-step inventory balance for snapshot NL OPF. Multi-hour coupled runs use
:func:`~pyflow_acdc.window_nl_opf` (parent ``window_h2_constraints``; terminal
``H2_mass_final`` when set).

Myopic :func:`~pyflow_acdc.ts_acdc_opf` **carries** H₂ inventory hour-to-hour
within ``H2_mass_max``. ``empty_tank_cycle`` (``None`` or ``N``) empties the
tank between solves (never / every ``N`` hours). ``H2_mass_final`` is not
enforced in myopic OPF; use ``H2_sale`` for economics. Rolling empties follow
the same attribute (every window vs boundaries at/past ``k·N``).

Production each frame (Useche-Arteaga et al. 2026 §3.4; ``c_h`` every hour):

.. math::

   h = b_h\, P_e\, S_{\mathrm{base}}\, \Delta t + c_h

   M_t = M_{t-1} + h

The formulation follows Useche-Arteaga et al. (2026) [#useche2026]_.
See :doc:`../citing` for the BibTeX entry.

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
