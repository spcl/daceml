from dace.transformation import pattern_matching

from .constant_folding import ConstantFolding
pattern_matching.Transformation.register(ConstantFolding, singlestate=True)
