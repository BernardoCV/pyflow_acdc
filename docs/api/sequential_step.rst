Sequential STEP
===============

Runs transmission expansion **one investment period at a time**: each step is a
single static :func:`transmission_expansion` (or, for ``sequential_MS_STEP``, a
:func:`multi_scenario_TEP`) solve, with the grid state carried forward to the
next period. This is an alternative to the multi-period formulations
(:func:`multi_period_transmission_expansion`, :func:`multi_period_MS_TEP`) when
you prefer a series of linked single-expansion solves over one large MP model.

Requires OPF/pyomo (``pip install pyflow-acdc[opf]``).

Functions are found in ``pyflow_acdc.ACDC_sequential_STEP``.

.. autofunction:: pyflow_acdc.sequential_STEP

.. autofunction:: pyflow_acdc.sequential_MS_STEP
