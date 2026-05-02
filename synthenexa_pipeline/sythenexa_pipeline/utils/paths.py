"""Resolve paths so `Dataset/...` always lives under sythenexa_pipeline/, not the repo root."""

from __future__ import annotations

from pathlib import Path

# sythenexa_pipeline/ (parent of utils/)
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent


def pipeline_root() -> Path:
    return _PIPELINE_ROOT


def dataset_dir() -> Path:
    return _PIPELINE_ROOT / "Dataset"


def resolve_pipeline_path(path: str | Path) -> Path:
    """
    Relative paths whose first segment is `Dataset` resolve under sythenexa_pipeline,
    regardless of current working directory (fixes outputs going to repo-root Dataset/).
    Leading `./` is stripped. Absolute paths are returned resolved unchanged.
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()

    parts = list(p.parts)
    while parts and parts[0] == ".":
        parts = parts[1:]
    if not parts:
        return (Path.cwd() / p).resolve()

    if parts[0] == "Dataset":
        return (_PIPELINE_ROOT.joinpath(*parts)).resolve()
    return (Path.cwd() / p).resolve()
