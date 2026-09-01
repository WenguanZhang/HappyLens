"""Unavailable CDWDN loss placeholder; see the DWDN upstream repository."""

UPSTREAM_URL = "https://github.com/dongjxjx/dwdn"


class Loss:
    """Placeholder that reports why the CDWDN loss is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The CDWDN loss is not distributed with HappyLens because CDWDN was "
            "adapted from DWDN, whose upstream repository does not provide an "
            f"explicit redistribution license: {UPSTREAM_URL}"
        )
