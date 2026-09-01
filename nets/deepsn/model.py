"""Unavailable DeepSN-Net model placeholder.

The adapted implementation is intentionally not distributed because the
upstream repository does not provide an explicit redistribution license.
"""

UPSTREAM_URL = "https://github.com/pandazcx/DeepSN-Net"


class DEEPSN:
    """Placeholder that reports why DeepSN-Net is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DeepSN-Net is not distributed with HappyLens because its upstream "
            f"repository does not provide an explicit redistribution license: {UPSTREAM_URL}"
        )
