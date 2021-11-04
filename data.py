import torch
from transformers import AutoTokenizer
from transformers.data.processors.squad import *
import os
import pandas as pd

import pickle5 as pickle

from torch.utils.data import Dataset


class CorpusSC(Dataset):
    def __init__(
        self, path, file, model_name="xlm-roberta-base", local_files_only=False
    ):
        self.max_sequence_length = 128
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = 0
        self.sequence_a_segment_id = 0
        self.sequence_b_segment_id = 1
        self.pad_token_segment_id = 0
        self.cls_token_segment_id = 0
        self.mask_padding_with_zero = True
        self.doc_stride = 128
        self.max_query_length = 64

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=False, local_files_only=local_files_only
        )

        self.label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}

        cached_data_file = path + f"_{type(self.tokenizer).__name__}.pickle"

        if os.path.exists(cached_data_file):
            self.data = pickle.load(open(cached_data_file, "rb"))
        else:
            self.data = self.preprocess(path, file)
            with open(cached_data_file, "wb") as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def preprocess(self, path, file):
        header = ["premise", "hypothesis", "label"]
        df = pd.read_csv(path, sep="\t", header=None, names=header)

        premise_list = df["premise"].to_list()
        hypothesis_list = df["hypothesis"].to_list()
        label_list = df["label"].to_list()

        # Tokenize input pair sentences
        ids = self.tokenizer(
            premise_list,
            hypothesis_list,
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        input_ids = ids["input_ids"]
        attention_mask = ids["attention_mask"].bool()
        token_type_ids = ids["token_type_ids"].bool()

        labels = torch.tensor(
            [self.label_dict[label] for label in label_list], dtype=torch.uint8
        )

        dataset = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": labels,
        }

        return dataset

    def __len__(self):
        return self.data["input_ids"].shape[0]

    def __getitem__(self, id):
        return {
            "input_ids": self.data["input_ids"][id],
            "attention_mask": self.data["attention_mask"][id],
            "token_type_ids": self.data["token_type_ids"][id],
            "label": self.data["label"][id],
        }
