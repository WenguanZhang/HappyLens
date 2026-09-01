"""Unavailable MIMO-UNet loss placeholder; see the upstream repository."""

UPSTREAM_URL = "https://github.com/chosj95/mimo-unet"


class Loss:
    """Placeholder that reports why the MIMO-UNet loss is unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The MIMO-UNet loss is not distributed with HappyLens because its "
            "upstream repository does not provide an explicit redistribution "
            f"license: {UPSTREAM_URL}"
        )
