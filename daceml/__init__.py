import os
import logging

import daceml.transformation  # this registers the transformations with dace

if "DACEML_LOG_LEVEL" in os.environ:
    logging.basicConfig(
        level=getattr(logging, os.environ["DACEML_LOG_LEVEL"].upper()))
