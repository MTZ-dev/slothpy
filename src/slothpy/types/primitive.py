from __future__ import annotations

from typing import Annotated

from pydantic import Field, TypeAdapter

# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

type PositiveInt = Annotated[int, Field(strict=True, gt=0)]

positive_int_adapter: TypeAdapter[PositiveInt] = TypeAdapter(PositiveInt)

__all__ = [
    "PositiveInt",
    "positive_int_adapter",
]
