"""Unavailable DeepSN-Net loss placeholder; see the upstream repository."""

UPSTREAM_URL = "https://github.com/pandazcx/DeepSN-Net"


class Loss:
    """Placeholder that reports why the DeepSN-Net loss is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The DeepSN-Net loss is not distributed with HappyLens because its "
            "upstream repository does not provide an explicit redistribution "
            f"license: {UPSTREAM_URL}"
        )
