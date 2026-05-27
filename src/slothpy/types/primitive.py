from __future__ import annotations

from typing import Annotated

from pydantic import Field, TypeAdapter

# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

type PositiveInt = Annotated[int, Field(strict=True, gt=0)]

type NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

positive_int_adapter: TypeAdapter[PositiveInt] = TypeAdapter(PositiveInt)
non_negative_int_adapter: TypeAdapter[NonNegativeInt] = TypeAdapter(NonNegativeInt)

__all__ = [
    "NonNegativeInt",
    "PositiveInt",
    "non_negative_int_adapter",
    "positive_int_adapter",
]
