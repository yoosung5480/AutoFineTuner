from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange  #for progress bars
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image #for image rendering
import re

MAX_LEN = 512
class BertDataset:
    def __init__(self, texts, labels=None, tokenizer=None, apply_preprocess:bool=True, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.apply_preprocess = apply_preprocess
    
    def preprocess_texts(self) -> tuple[list[str], list[int]]:
        """
        텍스트를 self.max_len 기준으로 문자열 길이로 분할 (prefix 없음 버전).
        "." 또는 "\n" 기준으로 문장을 나눈 뒤, 최대 길이에 도달할 때까지 쌓아서 하나의 조각으로 저장.
        """
        preprocessed_texts = []
        preprocessed_labels = []
        estimated_char_limit = int(self.max_len * 1.9) # 1000개 정도의 샘플에 적용해본결과 대략 1.9가 나온다.

        for i in range(len(self.texts)):
            text = self.texts[i]
            label = self.labels[i]

            # 길이 충분히 짧으면 그대로 사용
            if len(text) <= estimated_char_limit:
                preprocessed_texts.append(text.strip())
                preprocessed_labels.append(label)
                continue

            # 긴 텍스트는 문장 단위로 나눠서 조각 분할
            sentences = [s.strip() for s in re.split(r'[.\n]', text) if s.strip()]
            temp = ""
            
            for sentence in sentences:
                if len(temp) + len(sentence) + 1 <= estimated_char_limit:
                    temp += sentence + " "
                else:
                    preprocessed_texts.append(temp.strip())
                    preprocessed_labels.append(label)
                    temp = sentence + " "  # 다음 블록 시작

            # 마지막 블록도 저장
            if temp:
                preprocessed_texts.append(temp.strip())
                preprocessed_labels.append(label)

        self.preprocessed_texts = preprocessed_texts
        self.preprocessed_labels = preprocessed_labels
        return preprocessed_texts, preprocessed_labels


    
    def __tokenize__(
        self, 
    ) :
        if self.apply_preprocess:
            texts, labels = self.preprocess_texts()
        else:
            texts, labels = self.texts, self.labels

        # 1. gpu를 활용해서 하고싶다.
        # 2. 문장이 길어도, 너~무길다.
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']
        return input_ids, attention_masks, torch.tensor(labels)


    def set_loaders(self, batch_size=32, split_ratio=0.9):
        input_ids, attention_masks, labels = self.__tokenize__()

        if split_ratio == 1.0:
            # 전체 데이터를 train으로 사용
            self.train_data = TensorDataset(input_ids, attention_masks, labels)
            self.valid_data = None

            self.train_loader = DataLoader(
                self.train_data, sampler=RandomSampler(self.train_data), batch_size=batch_size
            )
            self.valid_loader = None

        elif split_ratio == 0.0:
            # 전체 데이터를 valid로 사용
            self.train_data = None
            self.valid_data = TensorDataset(input_ids, attention_masks, labels)

            self.train_loader = None
            self.valid_loader = DataLoader(
                self.valid_data, sampler=SequentialSampler(self.valid_data), batch_size=batch_size
            )

        else:
            # Train/Valid split
            train_inputs, val_inputs, train_labels, val_labels = train_test_split(
                input_ids, labels, test_size=(1 - split_ratio), random_state=42
            )
            train_masks, val_masks = train_test_split(
                attention_masks, test_size=(1 - split_ratio), random_state=42
            )

            self.train_data = TensorDataset(train_inputs, train_masks, train_labels)
            self.valid_data = TensorDataset(val_inputs, val_masks, val_labels)

            self.train_loader = DataLoader(
                self.train_data, sampler=RandomSampler(self.train_data), batch_size=batch_size
            )
            self.valid_loader = DataLoader(
                self.valid_data, sampler=SequentialSampler(self.valid_data), batch_size=batch_size
            )


    def get_loaders(self):
        return self.train_loader, self.valid_loader
    
    # def get_test_loader(self, texts: list[str], batch_size: int = 32) -> DataLoader:
    #     """
    #     테스트용 데이터 로더 반환 (label 없이)
    #     """
    #     # 텍스트 전처리
    #     processed_texts = []
    #     estimated_char_limit = int(self.max_len * 1.9)

    #     for text in texts:
    #         if len(text) <= estimated_char_limit:
    #             processed_texts.append(text.strip())
    #             continue

    #         sentences = [s.strip() for s in re.split(r'[.\n]', text) if s.strip()]
    #         temp = ""
    #         for sentence in sentences:
    #             if len(temp) + len(sentence) + 1 <= estimated_char_limit:
    #                 temp += sentence + " "
    #             else:
    #                 processed_texts.append(temp.strip())
    #                 temp = sentence + " "
    #         if temp:
    #             processed_texts.append(temp.strip())

    #     # 토큰화
    #     encodings = self.tokenizer.batch_encode_plus(
    #         processed_texts,
    #         add_special_tokens=True,
    #         truncation=True,
    #         padding='max_length',
    #         max_length=self.max_len,
    #         return_attention_mask=True,
    #         return_tensors='pt'
    #     )

    #     input_ids = encodings['input_ids']
    #     attention_masks = encodings['attention_mask']

    #     # DataLoader
    #     test_data = TensorDataset(input_ids, attention_masks)
    #     test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

    #     return test_loader
    
#### 현재 csv 문제상황을 위한 특수 csv처리 코드
def from_csv(path: str) -> tuple[list[str], list[int]]:
    """
    CSV에서 full_text, label 함께 반환
    """
    df = pd.read_csv(path).sample(100)
    # texts = ("제목 : " + df['title'].astype(str) + ", " + df['full_text'].astype(str)).tolist()
    texts = df['full_text'].astype(str).to_list()
    labels = df['generated'].astype(int).tolist()
    return texts, labels
