import numpy as np
from llm_sdk import Small_LLM_Model
from models import FunctionDefinition
from typing import Dict, List
import json


class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._load_vocab()
        self.reverse_vocab = self.token_to_id

    def _load_vocab(self):
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        self.token_to_id = raw_vocab
        self.id_to_token = {int(v): k for k, v in raw_vocab.items()}

    def _get_allowed_token_ids(self,
                               current_input_ids: List[int],
                               functions: List[FunctionDefinition]
                               ) -> List[int]:
        """Analyse l'état actuel pour restreindre les choix de l'IA."""
        current_text = self.model.decode(current_input_ids)
        allowed_ids = []
        if "{" not in current_text:
            token_bracket = "{"
            if token_bracket in self.reverse_vocab:
                allowed_ids.append(self.reverse_vocab[token_bracket])
            return allowed_ids
        if current_text.endswith('"name": "'):
            for fn in functions:
                if fn.name in self.reverse_vocab:
                    allowed_ids.append(self.reverse_vocab[fn.name])
            return allowed_ids
        return list(self.id_to_token.keys())

    def generate(self, prompt: str,
                 functions: List[FunctionDefinition]) -> str:
        raw_ids = self.model.encode(prompt)
        if hasattr(raw_ids, "tolist"):
            input_ids = raw_ids.tolist()
        else:
            input_ids = raw_ids
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids = [int(i) for i in input_ids]
        prompt_length = len(input_ids)

        while True:
            logits_list = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(logits_list)
            allowed_tokens = self._get_allowed_token_ids(input_ids, functions)
            mask = np.full(logits.shape, -float('inf'))
            if allowed_tokens:
                mask[allowed_tokens] = 0
            next_token_id = int(np.argmax(logits + mask))
            input_ids.append(next_token_id)
            last_char = self.model.decode([next_token_id])
            if "}" in last_char:
                break
            if len(input_ids) > 500:
                break

        return self.model.decode(input_ids[prompt_length:])
