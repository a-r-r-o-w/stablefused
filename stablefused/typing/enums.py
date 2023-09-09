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


class Scheduler(str, Enum):
    DEIS = "deis"
    DDIM = "ddim"
    DDPM = "ddpm"
    DPM2_KARRAS = "dpm2_karras"
    DPM2_KARRAS_ANCESTRAL = "dpm2_karras_ancestral"
    DPM_SDE = "dpm_sde"
    DPM_SDE_KARRAS = "dpm_sde_karras"
    DPM_MULTISTEP = "dpm_multistep"
    DPM_MULTISTEP_KARRAS = "dpm_multistep_karras"
    DPM_SINGLESTEP = "dpm_singlestep"
    DPM_SINGLESTEP_KARRAS = "dpm_singlestep_karras"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    HEUN = "heun"
    LINEAR_MULTISTEP = "linear_multistep"
    PNDM = "pndm"
    UNIPC = "unipc"
