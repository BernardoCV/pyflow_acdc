"""Resolve ``examples/`` CSV paths for git checkouts and pip installs."""
from __future__ import annotations

import sys
from pathlib import Path


def _is_url(path) -> bool:
    text = str(path)
    return text.startswith("http://") or text.startswith("https://")


def example_data_dirs(example_subdir: str, *, repo_root: Path):
    """Yield candidate local roots for an ``examples/<subdir>/`` tree."""
    yield repo_root / "examples" / example_subdir
    yield Path(sys.prefix) / "examples" / example_subdir


def resolve_example_data_path(
    filename,
    *,
    example_subdir: str,
    github_base: str,
    repo_root: Path,
    online: bool = True,
    relative_path: str | Path = "",
) -> str:
    """Return a local file path or a GitHub raw URL.

    Search order: git-checkout ``examples/``, pip ``{prefix}/examples/``,
    then ``github_base`` when ``online`` is True.
    """
    if _is_url(filename):
        return str(filename)

    name = Path(filename).name
    rel = Path(relative_path)
    for data_dir in example_data_dirs(example_subdir, repo_root=repo_root):
        path = data_dir / rel / name
        if path.is_file():
            return str(path)

    if online:
        rel_part = f"{rel.as_posix()}/" if rel.parts else ""
        return f"{github_base}{rel_part}{name}"

    rel_display = rel / name if rel.parts else name
    raise FileNotFoundError(
        f"Example data file not found: {rel_display} under examples/{example_subdir}/ "
        f"(online=False)."
    )


def first_example_data_dir(example_subdir: str, *, repo_root: Path) -> Path:
    """First existing local examples directory, else the git-checkout path."""
    dirs = list(example_data_dirs(example_subdir, repo_root=repo_root))
    for data_dir in dirs:
        if data_dir.is_dir():
            return data_dir
    return dirs[0]
