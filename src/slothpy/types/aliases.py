from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

type PathLike = str | Path
type NodeKind = Literal["file", "group", "dataset", "coordinate", "data_variable"]
type XarrayChunks = int | str | dict[str, Any] | None
