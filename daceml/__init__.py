import os
import logging

import daceml.transformation  # this registers the transformations with dace
from dace import config

log = logging.getLogger(__name__)

if "DACEML_LOG_LEVEL" in os.environ:
    logging.basicConfig(
        level=getattr(logging, os.environ["DACEML_LOG_LEVEL"].upper()))

if "--use_fast_math" in config.Config.get("compiler", "cuda", "args"):
    # disable cuda fast math: this causes nans to appear in BERT
    new_value = config.Config.get("compiler", "cuda",
                                  "args").replace("--use_fast_math", "")

    config.Config.set("compiler", "cuda", "args", value=new_value)
    log.info("Removed '--use_fast_math' from cuda args.")
