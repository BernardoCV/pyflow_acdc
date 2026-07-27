# -*- coding: utf-8 -*-
"""Solve progress reports and Plotly figures for the Tests tab."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go


@dataclass
class StudyReport:
    kind: str  # 'pf_ac' | 'pf_dc' | 'pf_acdc' | 'opf'
    elapsed_s: float | None = None
    final_tol: float | None = None
    log: str = ""
    tracker: dict | None = None
    ac_iters: list[int] = field(default_factory=list)
    solver_stats: dict | None = None
    timing_info: dict | None = None

    def summary_lines(self) -> list[str]:
        lines = [f"Study: {self.kind}"]
        if self.elapsed_s is not None:
            lines.append(f"Elapsed: {self.elapsed_s:.4f} s")
        if self.final_tol is not None:
            lines.append(f"Final tolerance: {self.final_tol}")
        if self.ac_iters:
            lines.append(f"AC Newton iterations: {self.ac_iters}")
        if self.tracker:
            seq = self.tracker.get("sequential_iterations") or []
            lines.append(f"Sequential outer iters: {len(seq)}")
            if self.tracker.get("final_sequential_tolerance") is not None:
                lines.append(
                    f"Sequential final tol: {self.tracker['final_sequential_tolerance']}"
                )
        if self.solver_stats:
            lines.append(f"Solver: {self.solver_stats.get('solver')}")
            lines.append(f"Termination: {self.solver_stats.get('termination_condition')}")
            lines.append(f"Solution found: {self.solver_stats.get('solution_found')}")
        return lines


def figure_from_study_report(report: StudyReport | None) -> go.Figure:
    """Build an interactive Plotly figure for the last solve."""
    fig = go.Figure()
    if report is None:
        fig.update_layout(title="No solve yet", template="plotly_white")
        return fig

    if report.kind in ("pf_ac", "pf_dc") and report.ac_iters:
        fig.add_trace(
            go.Bar(
                x=[f"call {i+1}" for i in range(len(report.ac_iters))],
                y=report.ac_iters,
                name="Newton iterations",
            )
        )
        fig.update_layout(
            title=f"PF Newton iterations (tol={float(np.asarray(report.final_tol).reshape(-1)[0]):.3e})",
            yaxis_title="Iterations",
            template="plotly_white",
        )
        return fig

    if report.kind == "pf_acdc" and report.tracker:
        tr = report.tracker
        seq = list(tr.get("sequential_iterations") or [])
        ac = list(tr.get("ac_pf_tolerances") or [])
        dc = list(tr.get("dc_pf_tolerances") or [])
        n = max(len(seq), len(ac), len(dc), 1)
        x = list(range(1, n + 1))
        if seq:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(seq) + 1)),
                    y=seq,
                    mode="lines+markers",
                    name="Sequential |ΔP|",
                )
            )
        if ac:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(ac) + 1)),
                    y=ac,
                    mode="lines+markers",
                    name="AC PF tol",
                )
            )
        if dc:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(dc) + 1)),
                    y=dc,
                    mode="lines+markers",
                    name="DC PF tol",
                )
            )
        fig.update_layout(
            title="AC/DC sequential PF convergence",
            xaxis_title="Outer iteration",
            yaxis_title="Tolerance / mismatch",
            yaxis_type="log",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    if report.kind == "opf" and report.solver_stats:
        return figure_opf_feasibility(report.solver_stats)

    # Fallback summary card
    fig.add_annotation(
        text="<br>".join(report.summary_lines()),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#111111"),
        align="left",
    )
    fig.update_layout(
        title="Solve summary",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
    )
    return fig


def figure_opf_feasibility(
    solver_stats: dict,
    *,
    x_axis: str = "iterations",
    y_axis: str = "objective",
) -> go.Figure:
    """Plotly version of ``plot_model_feasibility`` (all_solutions)."""
    solutions = solver_stats.get("all_solutions") or []
    fig = go.Figure()
    if not solutions:
        fig.update_layout(
            title="OPF feasibility — no solver progress points",
            template="plotly_white",
        )
        return fig

    if x_axis == "time":
        x_data = [s[0] for s in solutions]
        x_title = "time"
    else:
        x_data = [s[2] for s in solutions]
        x_title = "iterations"

    if y_axis == "iterations":
        y_data = [s[2] for s in solutions]
        y_title = "iterations"
    else:
        y_data = [s[1] for s in solutions]
        y_title = "objective"

    feasible_x, feasible_y = [], []
    regular_x, regular_y = [], []
    for i, solution in enumerate(solutions):
        is_feasible = bool(solution[4]) if len(solution) > 4 else False
        if is_feasible:
            feasible_x.append(x_data[i])
            feasible_y.append(y_data[i])
        regular_x.append(x_data[i])
        regular_y.append(y_data[i])

    if regular_x:
        fig.add_trace(
            go.Scatter(
                x=regular_x,
                y=regular_y,
                mode="lines+markers",
                name="NLP progress",
                line=dict(color="blue"),
            )
        )
    if feasible_x:
        fig.add_trace(
            go.Scatter(
                x=feasible_x,
                y=feasible_y,
                mode="markers",
                name="Feasible",
                marker=dict(color="red", size=10),
            )
        )
    fig.update_layout(
        title="OPF solver progress / feasibility",
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig
