from abc import ABC, abstractmethod
from typing import Dict, Union, List
from transformers import PreTrainedTokenizer


class PromptBase(ABC):

    @abstractmethod
    def build_prompt_template(self,
                              example: Dict[str, str],
                              tokenizer: PreTrainedTokenizer,
                              tokenize: bool = True) -> Union[str, List[int]]:
        raise NotImplementedError
