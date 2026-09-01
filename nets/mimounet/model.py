"""Unavailable MIMO-UNet model placeholder.

The adapted implementation is intentionally not distributed because the
upstream repository does not provide an explicit redistribution license.
"""

UPSTREAM_URL = "https://github.com/chosj95/mimo-unet"


class MIMOUNET:
    """Placeholder that reports why MIMO-UNet is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "MIMO-UNet is not distributed with HappyLens because its upstream "
            f"repository does not provide an explicit redistribution license: {UPSTREAM_URL}"
        )
