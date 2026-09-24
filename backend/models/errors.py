"""Domain errors: the request is well-formed but physically or contractually impossible.

Only these map to HTTP 422 at the API boundary. Any other exception is a bug and must surface
as a 500 so it is noticed, not disguised as a user error.
"""

from __future__ import annotations


class DomainError(ValueError):
    """A user-supplied input cannot be acted on for a stated physical reason."""


class MissingParameterError(DomainError):
    """A nameplate parameter the calculation needs was not supplied. It is never defaulted."""

    def __init__(self, parameter: str, purpose: str) -> None:
        self.parameter = parameter
        super().__init__(f"{parameter} is required for {purpose}; it is not defaulted")
