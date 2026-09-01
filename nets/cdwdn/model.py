"""Unavailable CDWDN model placeholder.

CDWDN was adapted from DWDN. The implementation is intentionally not
distributed because DWDN does not provide an explicit redistribution license.
"""

UPSTREAM_URL = "https://github.com/dongjxjx/dwdn"


class CDWDN:
    """Placeholder that reports why CDWDN is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "CDWDN is not distributed with HappyLens because it was adapted from "
            "DWDN, whose upstream repository does not provide an explicit "
            f"redistribution license: {UPSTREAM_URL}"
        )
