import logging

import bentoml
from bentoml.io import JSON
from fastapi import FastAPI
from pydantic import BaseModel, conint, constr
from pydantic.typing import Dict, List

from code_completion.constants import PIPELINE_PREFIX, SERVICE_NAME, AllowedModels
from code_completion.bento_helper import get_model_runner

logging.basicConfig(level=logging.INFO)


#  input schema
class InputSchema(BaseModel):
    max_length: conint(ge=1, le=500)
    n_sequences: conint(ge=1, le=5)
    prompt: constr(min_length=1, max_length=250)
    selected_model: AllowedModels


# output schema
class GeneratedText(BaseModel):
    length: int
    max_length: int
    prompt: str
    model_output: str
    model: AllowedModels


class OutputSchema(BaseModel):
    doc_list: List[GeneratedText]


output_spec = JSON(pydantic_model=OutputSchema)
input_spec = JSON(pydantic_model=InputSchema)


MODEL_TO_RUNNER = {
    model.value: get_model_runner(model.value, PIPELINE_PREFIX) for model in AllowedModels
}

# service definition
service = bentoml.Service(
    SERVICE_NAME,
    runners=list(MODEL_TO_RUNNER.values()),
)


# service API definition
@service.api(
    route="/code-completion",
    input=input_spec,
    output=output_spec,
)
def completion(input_data: input_spec) -> OutputSchema:
    # get data from input
    max_length = input_data.max_length
    n_sequences = input_data.n_sequences
    prompt = input_data.prompt
    selected_model = input_data.selected_model

    # generate text
    runner = MODEL_TO_RUNNER.get(selected_model)
    generated_text = runner.run(
        prompt, max_length=max_length, num_return_sequences=n_sequences
    )

    # generate output
    doc_list = [
        GeneratedText(
            length=len(generated_text[i]),
            max_length=max_length,
            model=selected_model,
            prompt=prompt,
            model_output=generated_text[i]["generated_text"],
        )
        for i in range(n_sequences)
    ]

    return OutputSchema(doc_list=doc_list)


# add additional endpoints
fastapi_app = FastAPI()
service.mount_asgi_app(fastapi_app)


@fastapi_app.get("/models")
def metadata() -> Dict:
    return {"AllowedModels": [model.value for model in AllowedModels]}
