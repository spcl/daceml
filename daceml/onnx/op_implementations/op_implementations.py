import copy
from math import sqrt

import dace
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.symbolic import symstr

from daceml.onnx.implementation_repository import register_pure_expansion
