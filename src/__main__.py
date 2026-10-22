import json
import os
import argparse
from typing import List
from pydantic import TypeAdapter, ValidationError
from llm_sdk import Small_LLM_Model
from .models import FunctionTest, FunctionDefinition
from .processor import ConstrainedDecoder


def load_functions(path: str) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path (str): The file path to the JSON file containing function
        definitions.

    Returns:
        List[FunctionDefinition]: A list of validated FunctionDefinition
        objects.
    """
    if os.path.getsize(path) == 0:
        raise ValueError("JSON file is empty.")
    with open(path, "r") as f:
        try:
            raw_functions = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("JSON file is empty or corrupted.")
    if not raw_functions:
        raise ValueError("JSON file doesn't contain any data.")
    adapter = TypeAdapter(List[FunctionDefinition])
    functions = adapter.validate_python(raw_functions)
    return functions


def load_tests(path: str) -> List[FunctionTest]:
    """Load and validate function calling tests from a JSON file.

    Args:
        path (str): The file path to the JSON file containing test cases.

    Returns:
        List[FunctionTest]: A list of validated FunctionTest objects.

    Raises:
        ValueError: If the file is empty, corrupted, or contains invalid data.
    """
    if os.path.getsize(path) == 0:
        raise ValueError("JSON file is empty.")
    with open(path, "r") as f:
        try:
            raw_tests = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("JSON file is empty or corrupted.")
    if not raw_tests:
        raise ValueError("JSON file doesn't contain any data.")
    adapter_tests = TypeAdapter(List[FunctionTest])
    tests = adapter_tests.validate_python(raw_tests)
    for index, test in enumerate(tests):
        if not test.prompt.strip():
            raise ValueError(f"Prompt at position {index} is empty.")
    return tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check"
                                     " Function Calling's implementation")
    parser.add_argument("--functions_definition", type=str,
                        default="data/input/functions_definition.json",
                        help="Path of the file with functions' definitions")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json",
                        help="Path of the test's file")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json",
                        help="Output file path")
    error_flag = 0
    args = parser.parse_args()
    model = Small_LLM_Model()
    try:
        file = open(args.functions_definition, "r")
        file.close()
        file = open(args.input, "r")
        file.close()
    except FileNotFoundError as e:
        error_flag = 1
        print(f"ERROR: No file found at {e.filename}")
    if error_flag == 0:
        try:
            functions = load_functions(args.functions_definition)
        except ValueError as e:
            print(f"ERROR: {e}")
            error_flag = 1
        try:
            tests = load_tests(args.input)
        except ValidationError as e:
            error_flag = 1
            for error in e.errors():
                if error['type'] == 'missing' and 'prompt' in error['loc']:
                    print("ERROR : JSON Key empty or missing."
                          " Key 'prompt' is mandatory.")
                else:
                    print(f"ERROR Pydantic : {error['msg']}")
        except ValueError as e:
            error_flag = 1
            print(f"ERROR : {e}")
    if error_flag == 0:
        decoder: ConstrainedDecoder = ConstrainedDecoder(model)
        final_results = []
        for entry in tests:
            current_prompt_text = entry.prompt
            print(f"Treatement : {current_prompt_text}")
            output_obj = decoder.generate(current_prompt_text, functions)
            clean_entry = {
                "prompt": current_prompt_text,
                "name": output_obj.get("name"),
                "parameters": output_obj.get("parameters")
            }
            final_results.append(clean_entry)
        output_sp = args.output.split("/")
        output_sp.pop()
        dir_path = "/".join(output_sp)
        os.makedirs(dir_path, exist_ok=True)
        with open(f"{args.output}",
                  "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nFile generated successfully: {len(final_results)} results.")
