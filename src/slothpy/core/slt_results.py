from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isfunction
from typing import Any, ClassVar

import xarray as xr
from pydantic import ConfigDict, validate_call

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

SLT_RESULT_VIEW_REGISTRY: dict[str, type[SltResultView]] = {}

_VALIDATE_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
)


def _validate_public_method(method: Any) -> Any:
    validated = validate_call(
        config=_VALIDATE_CONFIG,
        validate_return=False,
    )(method)
    validated.__slothpy_validated__ = True
    return validated


class SltResultViewMeta(type):
    """
    Metaclass for :class:`SltResultView` subclasses.

    Responsibilities:
    - register classes by ``expected_slt_type``
    - apply Pydantic validation to public methods defined directly on the class
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        for attr_name, attr_value in list(namespace.items()):
            if attr_name.startswith("_"):
                continue

            if isinstance(attr_value, property):
                continue

            if isinstance(attr_value, staticmethod):
                func = attr_value.__func__
                if not getattr(func, "__slothpy_validated__", False):
                    namespace[attr_name] = staticmethod(_validate_public_method(func))
                continue

            if isinstance(attr_value, classmethod):
                func = attr_value.__func__
                if not getattr(func, "__slothpy_validated__", False):
                    namespace[attr_name] = classmethod(_validate_public_method(func))
                continue

            if isfunction(attr_value):
                if not getattr(attr_value, "__slothpy_validated__", False):
                    namespace[attr_name] = _validate_public_method(attr_value)

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        expected_slt_type = getattr(cls, "expected_slt_type", None)
        if isinstance(expected_slt_type, str):
            key = expected_slt_type.strip().upper()
            if key:
                SLT_RESULT_VIEW_REGISTRY[key] = cls  # type: ignore[assignment]

        return cls


@dataclass(frozen=True, slots=True)
class SltResults:
    """
    Bundle for writing one SlothPy semantic xarray group.

    Passed to :meth:`SltFile._write_slothpy_group` together with the target
    group name. Producers supply an ``xr.Dataset`` or ``DataArray``, optional
    ``slt_type`` and ``primary`` (stored as SlothPy dataset metadata), and
    optional extra entries applied to the returned :class:`SltGroup` attributes.
    """

    dataset: xr.Dataset | xr.DataArray
    slt_type: str | None = None
    primary: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_typed_slt_results(self) -> SltResultView:
        """
        Return the registered :class:`SltResultView` for this bundle's ``slt_type``.
        """
        return to_typed_slt_results(self)


@dataclass(frozen=True, slots=True)
class SltResultView(metaclass=SltResultViewMeta):
    """
    Typed view over in-memory :class:`SltResults`.

    Subclasses define:

        expected_slt_type: ClassVar[str] = "GROUP_TYPE_NAME"

    and are registered in :data:`SLT_RESULT_VIEW_REGISTRY`.
    """

    results: SltResults

    expected_slt_type: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        expected = self.expected_slt_type
        if expected is None:
            return

        slt_type = self.results.slt_type
        if slt_type is None:
            raise TypeError(
                f"Cannot construct {type(self).__name__} from results without slt_type; "
                f"expected {expected!r}."
            )

        if slt_type.strip().upper() != expected.strip().upper():
            raise TypeError(
                f"Cannot construct {type(self).__name__} from "
                f"slt_type={slt_type!r}; expected {expected!r}."
            )

    @property
    def dataset(self) -> xr.Dataset | xr.DataArray:
        return self.results.dataset

    @property
    def slt_type(self) -> str | None:
        return self.results.slt_type

    @property
    def primary(self) -> str | None:
        return self.results.primary

    @property
    def attrs(self) -> dict[str, Any]:
        return self.results.attrs


def to_typed_slt_results(results: SltResults) -> SltResultView:
    """
    Construct the registered :class:`SltResultView` for ``results.slt_type``.
    """
    slt_type = results.slt_type
    if slt_type is None:
        raise TypeError("Cannot wrap SltResults without slt_type.")

    key = slt_type.strip().upper()
    try:
        view_cls = SLT_RESULT_VIEW_REGISTRY[key]
    except KeyError:
        raise TypeError(
            f"No registered SltResultView for slt_type={slt_type!r}."
        ) from None

    return view_cls(results)


__all__ = [
    "SLT_RESULT_VIEW_REGISTRY",
    "SltResultView",
    "SltResultViewMeta",
    "SltResults",
    "to_typed_slt_results",
]
