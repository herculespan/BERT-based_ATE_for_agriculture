import torch
from config import MAX_LENGTH
from config import LABEL_ALL_TOKENS
from config import BATCH_SIZE


def align_labels(texts, labels, tokenizer, labels_to_ids):
    tokenized_inputs = tokenizer(
        texts, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(
                    labels_to_ids[labels[word_idx]] if LABEL_ALL_TOKENS else -100
                )
            except Exception as e:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def align_word_ids(texts, tokenizer):
    tokenized_inputs = tokenizer(
        texts, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if LABEL_ALL_TOKENS else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids
