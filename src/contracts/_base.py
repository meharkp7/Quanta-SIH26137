"""Shared Pydantic base class for every contract object in this package.

Centralizing the model config means "frozen + no silent extra fields +
validate on construction" is a single decision made once, not something
each of the ten files could individually get slightly wrong.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Base class for all frozen, strict, versioned contract objects.

    - frozen=True: attribute reassignment raises. NOTE this is a shallow
      freeze -- see `ImmutableMap` in this module for why plain dict-typed
      fields still need their own protection.
    - extra="forbid": an unexpected field is a validation error, not a
      silently-ignored typo. This matters once six people are producing
      these objects from different modules.
    - populate_by_name / use_enum_values kept off deliberately: we want
      enum members (e.g. ScopeAction.KEEP), not bare strings, once a model
      is constructed, even though enums here subclass str.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
