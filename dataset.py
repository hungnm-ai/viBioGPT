import re
import json
from typing import List, Dict, Union
from torch.utils.data import Dataset
from prompt.qa_prompt import QAPrompt
from transformers import PreTrainedTokenizer


class DataReader:
    def __init__(self, path: Union[str, List[str]]):

        if isinstance(path, str):
            path = [p.strip() for p in path.split(",")]
        self.path = path

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"Với câu hỏi “.*?”", "Để trả lời câu hỏi trên", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"bệnh viện Vinmec", "bệnh viện", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"hệ thống Y Khoa Vinmec", "bệnh viện", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"đặt hẹn qua tổng đài Vinmec", "thăm khám tại bệnh viện", text,
                      flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"edoctor", "AI-Doctor", text, flags=re.MULTILINE | re.IGNORECASE)
        return text

    def load_data(self) -> List[Dict[str, str]]:

        qa_pairs = []
        for path in self.path:
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    question = item['question']
                    answer = item['answer']
                    answer = self.clean_text(answer)
                    if "vinmec" not in question.lower():
                        answer = re.sub("vinmec", "AI-Doctor", answer,
                                        flags=re.MULTILINE | re.IGNORECASE)
                    answer = answer.strip()
                    question = question.strip()
                    if len(answer) == 0 or len(question) == 0:
                        continue
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
        print("Number of pairs: ", len(qa_pairs))
        # with open(os.path.join(app_path, "data", 'qa_medical_pairs.json'), 'w') as f:
        #     json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        return qa_pairs


class BioDataset(Dataset):
    def __init__(self,
                 examples: List,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024,
                 truncation: bool = True,
                 ignore_sample: bool = True,
                 sort_by_length: bool = False) -> None:
        qa_prompt = QAPrompt()
        dataset = []

        min_seq_len = max_length

        for i, example in enumerate(examples):
            prompt_ids = qa_prompt.build_prompt_template(example=example,
                                                         tokenizer=tokenizer,
                                                         tokenize=True,
                                                         max_length=max_length,
                                                         truncation=truncation)

            if i == 5:
                prompt_str = qa_prompt.build_prompt_template(example=example,
                                                             tokenizer=tokenizer,
                                                             tokenize=False,
                                                             max_length=max_length,
                                                             truncation=truncation)

                print("prompt_str: ", prompt_str)
                print("prompt_ids: ", prompt_ids)

            if (len(prompt_ids) > max_length or len(prompt_ids) < 128) and ignore_sample:
                continue

            dataset.append({
                "prompt_ids": prompt_ids,
                "length": len(prompt_ids)
            })

            if len(prompt_ids) < min_seq_len:
                min_seq_len = len(prompt_ids)
        if sort_by_length:
            dataset = sorted(dataset, key=lambda item: item['length'], reverse=True)
        self.dataset = [example["prompt_ids"] for example in dataset]

        print("Total dataset: ", len(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


if __name__ == '__main__':
    import os
    from utils import app_path

    paths = [os.path.join(app_path, 'data', 'qa_vinmec.json'),
             os.path.join(app_path, 'data', 'qa_pairs_edoctor.json')
             ]
    train_reader = DataReader(paths)
    data = train_reader.load_data()
    from tokenizer import load_tokenizer
    from arguments import TokenizerArgs

    _tokenizer = load_tokenizer(TokenizerArgs())
    bio_dataset = BioDataset(examples=data,
                             tokenizer=_tokenizer)
