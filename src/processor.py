import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionDefinition
from typing import Dict, List
import json


class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        self.vocab = self._load_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def _load_vocab(self) -> Dict[int, str]:
        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def _get_allowed_token_ids(self,
                               current_input_ids: List[int],
                               functions: List[FunctionDefinition]
                               ) -> List[int]:
        """
        Analyse les tokens actuels et retourne les IDs autorisés pour le
        suivant. C'est ici que tu implémentes ta machine à états.
        """
        current_text = self.model.decode(current_input_ids)
        allowed_ids = []
        if current_text.endswith('{"name": "'):
            for fn in functions:
                if fn.name in self.reverse_vocab:
                    allowed_ids.append(self.reverse_vocab[fn.name])
        return allowed_ids

    def generate(self, prompt: str, functions: List[FunctionDefinition]):
        input_ids = self.model.encode(prompt)
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_tokens = self._get_allowed_token_ids(input_ids)
            mask = np.full(logits.shape, -float('inf'))
            mask[allowed_tokens] = 0
            constrained_logits = logits + mask
            next_token_id = np.argmax(constrained_logits)
            input_ids.append(next_token_id)
            if self.is_eos(next_token_id):
                break
        return self.model.decode(input_ids)
