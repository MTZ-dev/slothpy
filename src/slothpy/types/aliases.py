from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# General purpose aliases
# ---------------------------------------------------------------------------

type PathLike = str | Path

# ---------------------------------------------------------------------------
# SltFile aliases
# ---------------------------------------------------------------------------

type NodeKind = Literal["file", "group", "dataset", "coordinate", "data_variable"]
type XarrayChunks = int | str | dict[str, Any] | None


# ---------------------------------------------------------------------------
# Hamiltonian aliases
# ---------------------------------------------------------------------------

type HamiltonianInteractionKind = Literal["SOC", "SOC_SSC"]
type HamiltonianRepresentationKind = Literal["CI", "DIAGONAL"]

# ---------------------------------------------------------------------------
# Array aliases
# ---------------------------------------------------------------------------

type ArrayOrder = Literal["C", "F"]
