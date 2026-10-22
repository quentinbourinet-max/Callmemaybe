import json
import numpy as np
from typing import Any, Dict, List
from torch import Tensor
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


class ConstrainedDecoder:
    """A decoder that generates constrained function call outputs using a
    language model.

    This class implements constrained decoding to ensure the generated text
    conforms to a specific JSON structure for function calling.
    """
    def __init__(self, model: Small_LLM_Model):
        """Initialize the ConstrainedDecoder with a language model.

        Args:
            model: The language model instance to use for generation.
        """
        self.model = model
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}

    def _clean_ids(self, raw_ids: Tensor) -> List[int]:
        """Clean and flatten raw token IDs into a list of integers.

        Args:
            raw_ids: Raw token IDs, potentially nested or as tensors.

        Returns:
            List[int]: A flattened list of integer token IDs.
        """
        if hasattr(raw_ids, "tolist"):
            ids = raw_ids.tolist()
        else:
            ids = list(raw_ids)
        while len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        return [int(i) for i in ids]

    def generate(self, prompt: str,
                 functions: List[FunctionDefinition]) -> Dict[Any, Any]:
        """Generate a constrained function call response based on the prompt
        and available functions.

        Args:
            prompt (str): The user prompt to generate a response for.
            functions (List[FunctionDefinition]): List of available function
            definitions.

        Returns:
            dict: A dictionary containing the prompt, function name, and
            parameters.
        """
        fn_schemas = []
        r = "\n"
        for f in functions:
            params_simple: dict[str, str] = {}
            for p in f.parameters:
                params_simple[p] = f"{f.parameters[p].type}"
            fn_schemas.append(f"- {f.name}: {params_simple}\n")
        system_prompt = (
            "SYSTEM: Strict JSON Generator.\n"
            f"AVAILABLE: {r.join(fn_schemas)}"
            "Follow EXACTLY the patern AVAILABLE\n"
            "No result allowed\n"
            "- For 'numbers', use regex '[0-9]+'.\n"
            "The regex for vowels is [aeiouAEIOU].\n"
            "If the prompt says 'with asterisks',"
            " the replacement string is '*'.\n"
            f"USER: {prompt}\n"
            "ASSISTANT: "
            )
        input_ids = self._clean_ids(self.model.encode(system_prompt))
        prompt_length = len(input_ids)
        input_ids.extend(self._clean_ids(self.model.encode('{"name":"')))
        transition_str = '","parameters":{'
        transition_ids = self._clean_ids(self.model.encode(transition_str))
        allowed_chars = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789+*":,.-_ {}[]\''
            )
        arg_tokens = [tid for tid, tstr in self.id_to_token.items()
                      if all(c in allowed_chars for c
                             in tstr.replace('Ġ', ' '))]
        in_params = False
        finished_name = False
        while True:
            logits = np.array(self.model.get_logits_from_input_ids(input_ids))
            mask = np.full(logits.shape, -float('inf'))
            curr = self.model.decode(input_ids[prompt_length:])
            allowed = []
            if not finished_name:
                name_so_far = curr.split('"')[-1]
                if any(fn.name == name_so_far for fn in functions):
                    allowed = [transition_ids[0]]
                    finished_name = True
                else:
                    for fn in functions:
                        if fn.name.startswith(name_so_far):
                            rem = fn.name[len(name_so_far):]
                            token_rem = self._clean_ids(self.model.encode(rem))
                            if token_rem:
                                allowed.append(token_rem[0])
            elif not in_params:
                if transition_str in curr:
                    in_params = True
                    allowed = arg_tokens
                else:
                    for i in range(len(transition_ids)):
                        if self.model.decode(transition_ids[:i+1]) not in curr:
                            allowed = [transition_ids[i]]
                            break
            else:
                if curr.count('{') > curr.count('}'):
                    allowed = list(arg_tokens)
                    if '}' in self.token_to_id:
                        allowed.append(self.token_to_id['}'])
                else:
                    if '}' in self.token_to_id:
                        allowed = [self.token_to_id['}']]
            allowed = [a for a in allowed if a is not None]
            if not allowed:
                allowed = [i for i in range(len(logits))]
            mask[allowed] = 0
            next_token_id = int(np.argmax(logits + mask))
            input_ids.append(next_token_id)
            res_text = self.model.decode(input_ids[prompt_length:])
            if res_text.count('}') >= 2:
                break
            if len(input_ids) > prompt_length + 200:
                break
        generated_json_str = self.model.decode(input_ids[prompt_length:])
        idx_end = generated_json_str.find('}',
                                          generated_json_str.find('}') + 1)
        clean_json_str = generated_json_str[:idx_end+1]
        try:
            data = json.loads(clean_json_str)
            params = data.get("parameters", {})
            for k, v in params.items():
                if isinstance(v, str):
                    if v.replace('.', '', 1).isdigit():
                        params[k] = float(v) if '.' in v else int(v)
            for v in params:
                if isinstance(params[v], int):
                    params[v] = float(params[v])
            return {
                "prompt": prompt,
                "name": data.get("name"),
                "parameters": params
            }
        except Exception:
            return {"prompt": prompt, "name": "error", "parameters": {}}
