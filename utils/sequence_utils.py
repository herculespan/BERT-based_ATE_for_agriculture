from torch.utils.data import Dataset
from utils.preprocessing_utils import align_labels
from config import MAX_LENGTH
import torch


class DataSequence(Dataset):
    def __init__(self, df, tokenizer, labels_to_ids):
        lb = [i.split() for i in df["labels"].values.tolist()]
        txt = df["text"].values.tolist()
        self.texts = [
            tokenizer(
                str(i),
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            for i in txt
        ]
        self.labels = [
            align_labels(i, j, tokenizer, labels_to_ids) for i, j in zip(txt, lb)
        ]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels
