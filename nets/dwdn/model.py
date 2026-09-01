"""Unavailable DWDN model placeholder.

The adapted implementation is intentionally not distributed because the
upstream repository does not provide an explicit redistribution license.
"""

UPSTREAM_URL = "https://github.com/dongjxjx/dwdn"


class DWDN:
    """Placeholder that reports why DWDN is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DWDN is not distributed with HappyLens because its upstream "
            f"repository does not provide an explicit redistribution license: {UPSTREAM_URL}"
        )
