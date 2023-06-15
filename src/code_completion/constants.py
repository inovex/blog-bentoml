from enum import Enum


# allowed models. simply add a new huggingface `CausalLM` and specify below if it needs sampling
class AllowedModels(str, Enum):
    codegen = "salesforce/codegen-350M-multi"
    codegen2 = "salesforce/codegen2-1b"
    codebert = "huggingface/CodeBERTa-small-v1"


# the huggingface pipeline task for which the model is prepared
PIPELINE_TASK = "text-generation"

PIPELINE_PREFIX = f"{PIPELINE_TASK}-pipeline-"
ALL_PIPELINES = [f"{PIPELINE_PREFIX}{model.value}" for model in AllowedModels]
SERVICE_NAME = "code_completion_service"
