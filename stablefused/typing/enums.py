from enum import Enum


class InpaintWalkType(str, Enum):
    """
    Enum for inpainting walk types.
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    FORWARD = "forward"
    BACKWARD = "backward"
