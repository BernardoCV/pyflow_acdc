"""Generate docs/api/cable_database.rst from bundled Cable_database YAML files."""

from __future__ import annotations

import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PKG_DIR = DOCS_DIR.parent / "pyflow_acdc"
CABLE_DIR = PKG_DIR / "Cable_database"
OUTPUT = DOCS_DIR / "api" / "cable_database.rst"

DISCLAIMER = (
    "PyFlow-ACDC takes **no ownership** of the cable parameters bundled in "
    "``pyflow_acdc/Cable_database/``. Data were obtained from the sources cited "
    "below and are included for **academic and testing purposes only**. "
    "Electrical parameters and especially ``Cost_per_km`` values do not "
    "represent commercial quotations or manufacturer warranties."
)


def _cable_name_and_reference(path: Path) -> tuple[str, str]:
    cable_name = None
    reference = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if cable_name is None and stripped.endswith(":") and not stripped.startswith("Reference"):
                cable_name = stripped[:-1]
                continue
            match = re.match(r"Reference:\s*(.+)", stripped)
            if match:
                reference = match.group(1).strip()
                break
    if not cable_name:
        raise ValueError(f"Could not read cable name from {path.name}")
    if not reference:
        raise ValueError(f"Missing Reference in {path.name}")
    return cable_name, reference


def _rst_escape(text: str) -> str:
    text = text.replace("\\", "\\\\")
    for char in "*|_[]":
        text = text.replace(char, f"\\{char}")
    return text


ORBIT_REFERENCE = "https://github.com/NLRWindSystems/ORBIT/tree/dev/library/cables"
ABB_REFERENCE = "ABB"

REFERENCE_OVERRIDES: dict[str, dict[str, str]] = {
    ABB_REFERENCE: {
        "display_name": "Aluminium test cables",
        "intro": (
            "These cables are used for Moray East and West tests in [1]_ using "
            "cable data from XLPE Submarine Cable Systems Attachment to XLPE "
            "Land Cable Systems and Power capacity from the projects "
            "documentation."
        ),
        "footnote": (
            ".. [1] Castro Valerio, B., Gebraad, P. M. O., Cheah-Mane, M., "
            "A. Lacerda, V., and Gomis-Bellmunt, O.: A multi-stage methodology "
            "for wind park inter-array cabling: graph preparation, layout, and "
            "sizing, Wind Energ. Sci. Discuss. [preprint], "
            "https://doi.org/10.5194/wes-2026-53, in review, 2026"
        ),
    },
}


def _reference_override(reference: str) -> dict[str, str] | None:
    return REFERENCE_OVERRIDES.get(reference)


def _reference_display_name(reference: str) -> str:
    override = _reference_override(reference)
    if override:
        return override["display_name"]
    if reference == ORBIT_REFERENCE:
        return "ORBIT"
    return reference


def _reference_section_lines(reference: str) -> list[str]:
    override = _reference_override(reference)
    if override:
        return [
            override["intro"],
            "",
            override["footnote"],
            "",
        ]
    return [
        "**Source.** {}".format(_reference_label(reference)),
        "",
    ]


def _reference_label(reference: str) -> str:
    if reference.startswith("http://") or reference.startswith("https://"):
        label = _reference_display_name(reference)
        return f"`{label} <{reference}>`__"
    return f"**{_rst_escape(reference)}**"


def _section_anchor(reference: str) -> str:
    slug = re.sub(
        r"[^a-z0-9]+", "-", _reference_display_name(reference).lower()
    ).strip("-")
    return slug or "reference"


def _collect_by_reference() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in sorted(CABLE_DIR.glob("*.yaml")):
        cable_name, reference = _cable_name_and_reference(path)
        grouped.setdefault(reference, []).append(cable_name)
    for cables in grouped.values():
        cables.sort()
    return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))


def generate(output: Path | None = None) -> Path:
    output = output or OUTPUT
    grouped = _collect_by_reference()
    total = sum(len(cables) for cables in grouped.values())

    lines = [
        ".. _cable_database:",
        "",
        "Cable Database",
        "==============",
        "",
        "Bundled AC/DC cable types loaded from ``pyflow_acdc/Cable_database/*.yaml`` "
        "when using ``Cable_type=...`` in :func:`~pyflow_acdc.add_line_AC`, "
        ":func:`~pyflow_acdc.add_line_DC`, and related helpers.",
        "",
        f"This page lists **{total}** cable entries in **{len(grouped)}** source "
        "groups (auto-generated from the YAML ``Reference`` field).",
        "",
        "Import and Extend",
        "-----------------",
        "",
        ".. autofunction:: pyflow_acdc.import_orbit_cables",
        "",
        ".. autofunction:: pyflow_acdc.expand_cable_database",
        "",
        ".. contents::",
        "   :local:",
        "   :depth: 1",
        "",
        "Summary",
        "-------",
        "",
        ".. warning::",
        "",
        "   " + DISCLAIMER,
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 60 10",
        "",
        "   * - Source (``Reference`` field)",
        "     - Cables",
    ]
    for reference, cables in grouped.items():
        display = _reference_display_name(reference)
        label = display if len(display) <= 80 else display[:77] + "..."
        lines.append(f"   * - {_rst_escape(label)}")
        lines.append(f"     - {len(cables)}")
    lines.append("")

    for reference, cables in grouped.items():
        anchor = _section_anchor(reference)
        display = _reference_display_name(reference)
        title = _rst_escape(display)
        lines.extend(
            [
                ".. _cable_ref_{}:".format(anchor),
                "",
                title,
                "~" * len(title),
                "",
                *_reference_section_lines(reference),
                f"**Cable types ({len(cables)}):**",
                "",
            ]
        )
        for cable_name in cables:
            lines.append(f"* ``{cable_name}``")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    path = generate()
    print(f"Wrote {path}")
