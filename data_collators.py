import numpy as np
from typing import *
from tokenizer import SpecialToken
from transformers import DataCollatorForLanguageModeling


class DataCollatorForCompletionLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        attention_mask = batch["labels"].clone()
        attention_mask[attention_mask != -100] = 1
        attention_mask[attention_mask == -100] = 0

        batch['attention_mask'] = attention_mask
        # The code then encodes a special token, RESPONSE_KEY_NL,
        # representing the end of the prompt followed by a newline.
        # It searches for this token in the sequence of tokens (labels)
        # and finds its index.
        # print(batch['attention_mask'])
        # print(batch['labels'])
        # print(batch['input_ids'])
        response_token_ids = self.tokenizer.encode(SpecialToken.end_instruct,
                                                   add_special_tokens=False)
        for i in range(len(examples)):
            label = batch["labels"][i]
            response_token_ids_start_idx = None
            for idx in np.where(label == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                response_token_ids_end_idx = -1
                # If the response token is not found in the sequence, it raises a RuntimeError.
                # Otherwise, it determines the end index of the response token.

                print(f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}')
                # raise RuntimeError(
                #     f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                # )
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + 1

            # To train the model to predict only the response and ignore the prompt tokens,
            # it sets the label values before the response token to -100.
            # This ensures that those tokens are ignored by the PyTorch loss function during training.
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels
        return batch
