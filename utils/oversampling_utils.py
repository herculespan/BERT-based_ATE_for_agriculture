import pandas as pd
from typing import List
import warnings
import numpy as np
import nltk
from nltk.corpus import wordnet
from utils.annotation_utils import get_lemma
from nltk.corpus import stopwords
from tqdm import tqdm

warnings.filterwarnings(action="once")
import spacy

nlp = spacy.load("en_core_web_sm")

ANNOTATION_RATIO_THRESHOLD = 0.3


def get_lemma(token: str) -> str:
    if len(token) > 1:
        doc = nlp(token)
        return doc[0].lemma_
    else:
        return token


def has_good_annotation_ratio(labels):
    return labels.count("B-Agr") / labels.count("O") > ANNOTATION_RATIO_THRESHOLD


def is_excluded_token(token: str, excluded_terms: List[str]) -> bool:
    """
    It returns whether a sentence contains a specific excluded term.
    """
    lemmatized_token = get_lemma(token)
    for excluded_term in excluded_terms:
        if excluded_term.casefold() == lemmatized_token.casefold():
            return True

    return False


def get_annotation_ratio(data):
    num_b_agri = 0
    num_i_agri = 0
    num_o = 0
    annotation_ratio = []
    for index, row in data.iterrows():
        annotation_ratio.append(row["labels"].count("B-Agr") / row["labels"].count("O"))
        num_b_agri += row["labels"].count("B-Agr")
        num_i_agri += row["labels"].count("I-Agr")
        num_o += row["labels"].count("O")

    return np.mean(annotation_ratio)


def get_synonyms(token: str) -> List[str]:
    synonyms = []

    for syn in wordnet.synsets(token):
        for lemma in syn.lemmas():
            if lemma.name().casefold() != token.casefold():
                synonyms.append(lemma.name())

    return list(set(synonyms))


def get_synonym(token: str) -> str:
    synonyms = []
    for syn in wordnet.synsets(token):
        for lemma in syn.lemmas():
            # print(lemma.name())
            if (
                get_lemma(lemma.name().casefold()) != get_lemma(token.casefold())
                and not get_lemma(lemma.name().casefold())
                in get_lemma(token.casefold())
                and not get_lemma(token.casefold())
                in get_lemma(lemma.name().casefold())
            ):
                # print(lemma.name().casefold())
                # print(get_lemma(lemma.name().casefold()))
                # print(get_lemma(token.casefold()))
                if "_" in lemma.name():
                    return lemma.name().replace("_", "-")
                return lemma.name()
    return ""


def is_stop_word(token: str) -> bool:
    return token in stopwords.words("english") or token.lower() in stopwords.words(
        "english"
    )


def get_semantic_equivalent_sentence(sentence: str, excluded_terms: List[str]) -> str:
    num_replacements = 0
    for token in sentence.split(" "):
        if (
            not is_excluded_token(token, excluded_terms)
            and not is_stop_word(token)
            and not token.isnumeric()
        ):
            synonym = get_synonym(token)
            # print(synonym)
            if (
                not is_excluded_token(synonym, excluded_terms)
                and not is_stop_word(synonym)
                and len(synonym) > 1
            ):
                sentence = sentence.replace(token, synonym)
                num_replacements += 1
    print(f"Num. replacements: {num_replacements}")
    return sentence


def get_oversampled_data_with_synonyms(
    data, excluded_terms, use_undersampling, verbose=False
):
    oversampled_dataset = data.copy(deep=True)
    sentences = []
    labels = []
    for index, row in tqdm(data.iterrows()):
        if has_good_annotation_ratio(row["labels"]):
            sentence = row["text"]
            equivalent_sentence = get_semantic_equivalent_sentence(
                sentence, excluded_terms
            )
            if equivalent_sentence != sentence:
                if verbose:
                    print(equivalent_sentence)
                    print(row["text"])
                sentences.append(equivalent_sentence)
                labels.append(row["labels"])
            if use_undersampling:
                sentences.append(sentence)
                labels.append(row["labels"])

    if not use_undersampling:
        oversampled_dataset = pd.concat(
            [
                oversampled_dataset,
                pd.DataFrame.from_dict({"text": sentences, "labels": labels}),
            ]
        )
    else:
        oversampled_dataset = pd.DataFrame.from_dict(
            {"text": sentences, "labels": labels}
        )

    return oversampled_dataset.sample(frac=1).reset_index(drop=True)
