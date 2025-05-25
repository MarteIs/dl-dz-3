from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import random

class DataProcessor:
    def __init__(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained)
        self.cfg = cfg

    def process_text(self, text: str):
        enc = self.tokenizer(text, truncation=True, max_length=self.cfg.seq_len)
        return {
            'input_ids': torch.tensor(enc['input_ids']),
            'attention_mask': torch.tensor(enc['attention_mask']),
            'text': text
        }