import logging
from pprint import pformat
from typing import Optional

import bentoml
from bentoml._internal.runner import Runner
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from code_completion.constants import AllowedModels

logger = logging.getLogger(__name__)


def load_models_into_bento(
    AllowedModels: type[AllowedModels],
    PIPELINE_TASK: str,
    PIPELINE_PREFIX: str,
    path: Optional[str] = None,
) -> None:
    """
    Loads the models in a huggingface format and converts it into a BentoML model.
    Args:
        AllowedModels (str, Enum): Class of AllowedModels specifying the huggingface models that shall be used
        PIPELINE_TASK (str): Task the model shall execute
        PIPELINE_PREFIX (str): Constant for naming the pipeline
        path (str): The path to the model to be loaded
    """
    for model in AllowedModels:
        model_name = model.value
        model_identifier = f"{path}/{model_name}" if path else model_name

        # load model into bento
        model = AutoModelForCausalLM.from_pretrained(
            model_identifier, trust_remote_code=True, revision="main"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)

        generator = pipeline(PIPELINE_TASK, model=model, tokenizer=tokenizer)
        bentoml.transformers.save_model(
            f'{PIPELINE_PREFIX}{model_name.replace("/", "-")}', generator
        )

    bento_models_list = bentoml.models.list()
    logger.info(pformat(bento_models_list))
    logger.info("Successfully loaded models into bento!")


# always
def get_model_runner(model_value: str, PIPELINE_PREFIX: str) -> Runner:
    """
    Get the newest model from the bentoml model store.
    Args:
        model_value (str): Value from AllowedModels specifying the name of the HuggingFace model.
        PIPELINE_PREFIX (str): Constant for naming the pipeline
    """
    # replace occurrences of "/" with "-" otherwise bentoml will not accept the name
    model_name = model_value.replace("/", "-")
    pipeline_name = f"{PIPELINE_PREFIX}{model_name}:latest"
    pipeline_model = bentoml.transformers.get(pipeline_name)
    return pipeline_model.with_options(kwargs={"trust_remote_code": True}).to_runner()
