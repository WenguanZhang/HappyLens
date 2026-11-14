from .optim import Merit, MeritZ


class Generation_Prime(Merit):
    """Shared merit interface for prime-lens structure generation."""


class Generation_Zoom(MeritZ):
    """Shared merit interface for zoom-lens structure generation."""
