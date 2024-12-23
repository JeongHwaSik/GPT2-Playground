import torch
import json
import tiktoken
import torch.nn as nn
from sklearn.model_selection import train_test_split

from model import GPT2
from config import GPTConfig


class NewsDataLoader:
    """
    News Category Dataset Loader
    Downloaded from here: https://www.kaggle.com/datasets/rmisra/news-category-dataset/data
    """

    def __init__(self, data_dir="./document_classification/data/News_Category_Dataset_v3.json", batch_size=32, device="cpu"):

        # load pre-trained GPT-2 decoder
        config = GPTConfig()
        self.embedding = GPT2(config).from_pretrained("gpt2")
        self.embedding.lm_head = nn.Identity()  # remove last lm_head projection
        self.embedding = self.embedding.to(device)

        total_dataset = []
        # load dataset & get all unique words in the category
        with open(data_dir) as f:
            for data in f:
                total_dataset.append(json.loads(data))

        # GPT2 Tokenizer to encode text
        tokenizer = tiktoken.get_encoding("gpt2")  # vocab_size=50257
        x = [tokenizer.encode(f"{d['headline']} {d['short_description']}".lower()) for d in total_dataset]
        y = [d['category'] for d in total_dataset]

        self.cat2idx = self._get_cat2idx(set(y))  # category to idx

        # train & val split
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

        self.x_train = x_train
        self.y_train = torch.tensor([self.cat2idx[cat] for cat in y_train])
        self.x_val = x_val
        self.y_val = torch.tensor([self.cat2idx[cat] for cat in y_val])

        self.num_classes = len(self.cat2idx)
        self.batch_size = batch_size
        self.device = device

        print(f"Training Samples: {len(self.x_train)}")
        print(f"Validation Samples: {len(self.x_val)}")
        print(f"Categories: {len(self.cat2idx)}")

        self.train_start_idx = 0
        self.val_start_idx = 0

    def gpt2_embedding(self, x):
        x_train = []
        for text in x:
            out, _ = self.embedding(torch.tensor(text).unsqueeze(0).to(self.device))
            x_train.append(out.mean(dim=1))  # average along length dimension
        x_train = torch.cat(x_train, dim=0)
        return x_train

    @staticmethod
    def _padding(tokens, max_padding):
        x = []
        for token in tokens:
            padded = torch.cat([torch.tensor(token), torch.zeros(max_padding - len(token)).fill_(50257)], dim=-1)
            x.append(padded)
        return torch.stack(x, dim=0)

    @staticmethod
    def _get_cat2idx(category):
        unique_cat = list(set(category))
        cat2idx = {}
        for idx, cat in enumerate(unique_cat):
            cat2idx[cat] = idx
        return cat2idx

    def get_next_batch(self):
        x = self.gpt2_embedding(self.x_train[self.train_start_idx:self.train_start_idx + self.batch_size])
        y = self.y_train[self.train_start_idx:self.train_start_idx + self.batch_size]
        self.train_start_idx += self.batch_size

        if self.train_start_idx >= len(self.x_train):
            self.train_start_idx = 0
        return x, y

    def get_next_val_batch(self):
        if self.val_start_idx + self.batch_size >= len(self.x_val):
            x = self.gpt2_embedding(self.x_val[self.val_start_idx:])
            y = self.y_val[self.val_start_idx:]
            self.val_start_idx = 0
        else:
            x = self.gpt2_embedding(self.x_val[self.val_start_idx:self.val_start_idx + self.batch_size])
            y = self.y_val[self.val_start_idx:self.val_start_idx + self.batch_size]
            self.val_start_idx += self.batch_size
        return x, y