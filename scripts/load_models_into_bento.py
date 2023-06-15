import logging

from code_completion.bento_helper import load_models_into_bento
from code_completion.constants import PIPELINE_PREFIX, PIPELINE_TASK, AllowedModels

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    load_models_into_bento(AllowedModels, PIPELINE_TASK, PIPELINE_PREFIX)
