"""A real immutable mapping type for use inside frozen contract models.

The V1 review flagged that a `frozen=True` dataclass/model containing a
plain `dict` field is not actually immutable: `model.metadata["x"] = "y"`
still mutates the dict in place even though reassigning `model.metadata`
itself is blocked. This module closes that gap with a thin
`types.MappingProxyType` wrapper that Pydantic knows how to validate,
coerce into, and serialize back out of.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class ImmutableStrMap(Mapping[str, str]):
    """A hashable, read-only string-to-string mapping.

    Accepts any `Mapping[str, str]` (including a plain dict) at
    construction/validation time, then exposes it read-only from then on.
    Two instances with equal contents compare and hash equal, which is what
    lets this sit safely inside a `frozen=True` model.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, str] | None = None) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(data or {})))

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImmutableStrMap):
            return dict(self._data) == dict(other._data)
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._data.items())))

    def __repr__(self) -> str:
        return f"ImmutableStrMap({dict(self._data)!r})"

    def to_dict(self) -> dict[str, str]:
        return dict(self._data)

    # -- Pydantic v2 integration -------------------------------------------------
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate(value: Any) -> "ImmutableStrMap":
            if isinstance(value, ImmutableStrMap):
                return value
            if isinstance(value, Mapping):
                for k, v in value.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        raise ValueError(
                            "ImmutableStrMap requires str keys and str values"
                        )
                return cls(value)
            raise TypeError(f"Cannot build ImmutableStrMap from {type(value)!r}")

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.to_dict()
            ),
        )
