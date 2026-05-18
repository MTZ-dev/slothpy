from __future__ import annotations

from inspect import isfunction
from typing import Any, ClassVar

from pydantic import ConfigDict, validate_call

from slothpy.core.slt import SltGroup

SLT_GROUP_TYPE_REGISTRY: dict[str, type[SltTypedGroup]] = {}

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


class SltTypedGroupMeta(type):
    """
    Metaclass for semantic SltGroup subclasses.

    Responsibilities:
    - automatically add __slots__ = () unless the class defines slots itself
    - register classes by expected_slt_type
    - apply Pydantic validation to public methods defined directly on the class
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        namespace.setdefault("__slots__", ())

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
                SLT_GROUP_TYPE_REGISTRY[key] = cls  # type: ignore[assignment]

        return cls


class SltTypedGroup(SltGroup, metaclass=SltTypedGroupMeta):
    """
    Base class for semantic SlothPy groups.

    Subclasses only need to define:

        expected_slt_type: ClassVar[str] = "GROUP_TYPE_NAME"

    The group is validated immediately after SltGroup initialization.
    """

    expected_slt_type: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_expected_slt_type()

    def _validate_expected_slt_type(self) -> None:
        expected = self.expected_slt_type

        if expected is None:
            return

        if not self.exists:
            raise KeyError(
                f"Group {self.path!r} does not exist in file {self.file_path!s}."
            )

        attrs = self.attrs.as_dict()
        group_type = attrs.get("slt_type")

        if isinstance(group_type, bytes):
            group_type = group_type.decode("utf-8")

        if not isinstance(group_type, str):
            raise TypeError(
                f"Group {self.path!r} is not a SlothPy semantic group: "
                "missing string attribute 'slt_type'."
            )

        expected_upper = expected.strip().upper()
        actual_upper = group_type.strip().upper()

        if actual_upper != expected_upper:
            raise TypeError(
                f"Group {self.path!r} has slt_type={group_type!r}, "
                f"expected {expected_upper!r}."
            )
