"""Unavailable DWDN loss placeholder; see the upstream repository."""

UPSTREAM_URL = "https://github.com/dongjxjx/dwdn"


class Loss:
    """Placeholder that reports why the DWDN loss is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The DWDN loss is not distributed with HappyLens because its upstream "
            f"repository does not provide an explicit redistribution license: {UPSTREAM_URL}"
        )
