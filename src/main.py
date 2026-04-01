import json
from typing import List
from pydantic import TypeAdapter
from llm_sdk import Small_LLM_Model
from models import FunctionTest, FunctionDefinition
from src.processor import ConstrainedDecoder


def load_functions() -> List[FunctionDefinition]:
    with open("data/input/functions_definition.json", "r") as f:
        raw_functions = json.load(f)
        adapter = TypeAdapter(List[FunctionDefinition])
        functions = adapter.validate_python(raw_functions)
    return functions


def load_tests() -> List[FunctionTest]:
    with open("data/input/function_calling_tests.json", "r") as f:
        raw_tests = json.load(f)
        adapter_tests = TypeAdapter(List[FunctionTest])
        tests = adapter_tests.validate_python(raw_tests)

    return tests


if __name__ == "__main__":
    model = Small_LLM_Model()
    functions = load_functions()
    tests = load_tests()
    decoder = ConstrainedDecoder(model)
    results = []
    for test in tests:
        print(f"Traitement du prompt : {test.prompt}")
        json_output = decoder.generate(test.prompt, functions)
        results.append(json_output)
        print(json_output)
    # 5. Sauvegarde finale (Étape 5)
    # ... code pour écrire dans data/output/ ...
    # print(functions)
    # print(tests)
