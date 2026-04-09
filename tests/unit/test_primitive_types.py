from __future__ import annotations

import pytest
from pydantic import ValidationError

from slothpy.types.primitive import positive_int_adapter


@pytest.mark.parametrize("value", [1, 2, 16])
def test_positive_int_accepts_positive_ints(value: int):
    assert positive_int_adapter.validate_python(value) == value


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_positive_int_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        positive_int_adapter.validate_python(value)
