import os
import logging

if "DACEML_LOG_LEVEL" in os.environ:
    logging.basicConfig(
        level=getattr(logging, os.environ["DACEML_LOG_LEVEL"].upper()))
