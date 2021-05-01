from dace.library import register_library

from .manual import LayerNorm, LayerNormEnvironment, DetectLN

register_library(__name__, "ort_ln")
