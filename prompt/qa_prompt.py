from prompt.role import Role
from constants import SYS_PROMPT
from prompt.base import PromptBase
from typing import Dict, Union, List
from transformers import PreTrainedTokenizer


class QAPrompt(PromptBase):
    def __init__(self, system_prompt: str = SYS_PROMPT):
        self.system_prompt = system_prompt

    def build_prompt_template(self,
                              example: Dict[str, str],
                              tokenizer: PreTrainedTokenizer,
                              tokenize: bool = True,
                              truncation: bool = True,
                              max_length: int = 1024) -> Union[str, List[int]]:
        conversation = [
            {
                "role": Role.system,
                "content": self.system_prompt
            },
            {
                "role": Role.user,
                "content": example['question']
            },
            {
                "role": Role.assistant,
                "content": example['answer']
            }
        ]
        prompt_str = tokenizer.apply_chat_template(conversation=conversation,
                                                   tokenize=tokenize,
                                                   truncation=truncation,
                                                   max_length=max_length)

        return prompt_str

    def build_prompt_instruction(self,
                                 question: str,
                                 tokenizer: PreTrainedTokenizer,
                                 tokenize: bool = False,
                                 truncation: bool = False,
                                 max_length: int = 1024) -> Union[str, List[int]]:
        conversation = [
            {
                "role": Role.system,
                "content": self.system_prompt
            },
            {
                "role": Role.user,
                "content": question
            }
        ]
        instruction_str = tokenizer.apply_chat_template(conversation=conversation,
                                                        tokenize=tokenize,
                                                        truncation=truncation,
                                                        max_length=max_length)

        return instruction_str
