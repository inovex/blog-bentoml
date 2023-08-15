# 🍱 BentoML

This is the repo for the blog post [BentoML for MLOps: From Prototype to Production](https://www.inovex.de/de/blog/bentoml-for-mlops-from-prototype-to-production/). For general infos on how to use BentoML check out [BentoML](https://docs.bentoml.org/en/latest/index.html).

<br>

## Prerequisites
### poetry
We use poetry as a dependency management tool. Set up your poetry environment with all needed dependencies installed by running:
```bash
poetry install
```
Then you can activate your poetry environment by running:
```bash
poetry shell
```
Alternatively you can use `poetry run` in front of commands to execute the command in the poetry environment.

Find more to poetry [here](https://python-poetry.org/docs/).

### pre-commit
Use pre-commit in your poetry environment or set it up in your default environment by running: 
```bash
pre-commit install
```

<br>

## 🚀 Getting started
1. Activate your poetry environment.
```bash
poetry shell
```
2. Load models from huggingface into your local BentoML model registry.
```bash
python scripts/load_models_into_bento.py
```
3. Build the bento. It will be saved at `~/bentoml/bentos/code_completion_service/` alongside the automatically generated Dockerfile and other files.
```bash
bentoml build
```
4. Now you can start the service (wait a few seconds till it has actually started).
```bash
bentoml serve code_completion_service:latest
```
5. You can check the UI at http://localhost:3000/, click on "Try it out" test it for yourself.

If you want to use a different model, add the model to the `AllowedModels` class in `constants.py` and rerun this pipeline.

<br>

## ✍ Usefull commands
Show local model registry
```bash
bentoml models list
```
Show local bentos
```bash
bentoml list
```
Build Docker image from generated Dockerfile
```bash
bentoml containerize code_completion_service:latest
```

Please refer to the [official BentoML docs](https://docs.bentoml.org/en/latest/) for more detailed information and tutorials.

<br>

## ☎️ Query the code generation API
Use the following Python commands to access the API:

```python
import requests

url = "http://0.0.0.0:3000/code-completion"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

data = {
    "max_length": 500,
    "n_sequences": 1,
    "prompt": "def fibonacci():",
    "selected_model": "salesforce/codegen-350M-multi",
}

response = requests.post(url, headers=headers, json=data)

# print the generated code
print(response.json()["doc_list"][0]["model_output"])
```
