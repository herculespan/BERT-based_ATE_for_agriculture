# import requests module
import multiprocessing
from typing import List, Dict
import time
from tqdm import tqdm
from langdetect import detect
import csv
import language_tool_python

tool = language_tool_python.LanguageTool("en-US")
import wordninja
import random
from nltk.corpus import stopwords

# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from config import MAX_GRAMMAR_WARNINGS
from config import ABSTRACTS_INPUT_FILE
from config import MIN_SENTENCE_WORDS
from config import MIN_CONCEPT_LENGTH
from config import INPUT_DATA_PATH
from config import OUTPUT_DATA_PATH
from config import PREF_LABELS_FILE
from config import ALT_LABELS_FILE

import spacy

nlp = spacy.load("en_core_web_sm")


def get_lemma(token: str) -> str:
    if len(token) > 1:
        doc = nlp(token)
        return doc[0].lemma_
    else:
        return token


def remove_mixed_words_from_sentence(sentence: str) -> str:
    """
    This function fixes words like "thissentence", by replacing them
    with the correct version (this sentence).
    """
    fixed_sentence = []
    sentence_tuples = [wordninja.split(word) for word in sentence.split(" ")]
    [fixed_sentence.extend(word_tuple) for word_tuple in sentence_tuples]
    return " ".join(fixed_sentence)


def contains_excluded_term(
    sentence: str, excluded_terms: List[str], is_strict: bool = False
) -> bool:
    """
    It returns whether a sentence contains a specific excluded term.
    """
    pure_sentence = [
        get_lemma(token.lower())
        for token in sentence.split(" ")
        if token.islower() or token.istitle()
    ]
    for excluded_term in excluded_terms:
        # print(excluded_term)
        if excluded_term not in stopwords.words("english"):
            if len(excluded_term) > 2:
                if (
                    excluded_term.islower() or excluded_term.istitle()
                ) and excluded_term.lower() in pure_sentence:
                    print(f"[INFO] Exluded Term Code 0 : {excluded_term}")
                    print()
                    return True
                elif " " in excluded_term and (
                    excluded_term.casefold() in sentence.casefold()
                    or excluded_term.casefold() in pure_sentence
                ):
                    print(f"[INFO] Exluded Term Code 1 : {excluded_term}")
                    return True
                elif (
                    excluded_term.isupper()
                    and len(excluded_term) > 2
                    and excluded_term in sentence
                ):
                    print(f"[INFO] Exluded Term Code 2 : {excluded_term}")
                    print(excluded_term)
                    return True
                if " " in excluded_term and is_strict:
                    for token in excluded_term.split(" "):
                        if (
                            token not in stopwords.words("english")
                            and get_lemma(token) in pure_sentence
                        ):
                            print(
                                f"[INFO] Exluded Term Code 3 : {excluded_term}, {token}"
                            )
                            return True

    print(f"[INFO] Any Exluded terms... Ready to annotate")
    return False


def remove_common_labels(pref_labels: List[str], alt_labels: List[str]) -> List[str]:
    """
    It removes those terms from alt_labels that appear also in pref_labels.
    """
    common_terms = list(set(pref_labels).intersection(set(alt_labels)))
    for common_term in common_terms:
        alt_labels.remove(common_term)


def annotate_sentence(sentence: str, ontology_subset: List[str]) -> List[str]:
    """
    Given a sentence, this function returns the IOB annotation format of it.
    """
    tokenizer = RegexpTokenizer(r"\w+")

    found_concepts = []
    annotated_sentence = ["O"] * len(sentence.split(" "))
    previous_token = "O"
    # ontology_subset = sorted(ontology_subset, key=len, reverse=True)

    # annotations = download_annotations(sentence)
    for concept in ontology_subset:
        if len(concept) >= MIN_CONCEPT_LENGTH and concept not in stopwords.words(
            "english"
        ):
            if concept.isupper():  # acronym
                occurrences = [
                    i for i in range(len(sentence)) if sentence.startswith(concept, i)
                ]
            else:
                occurrences = [
                    i
                    for i in range(len(sentence))
                    if sentence.lower().startswith(concept.lower(), i)
                ]
            for annotate_from in occurrences:
                found_concepts.append(concept)
                annotate_to = annotate_from + len(concept)
                annotation = {
                    get_lemma(token): ix
                    for ix, token in enumerate(
                        sentence[annotate_from:annotate_to].split(" ")
                    )
                }
                if len(annotation) == 1:
                    # for ix, token in enumerate(tokenizer.tokenize(sentence)):
                    for ix, token in enumerate(sentence.split(" ")):
                        lemma_token = get_lemma(token)
                        if annotation.get(lemma_token) is None:
                            previous_token = "O"
                        else:
                            if (
                                annotation[lemma_token] == 0
                                and annotated_sentence[ix] == "O"
                            ):
                                annotated_sentence[ix] = "B-Agr"
                                previous_token = "B-Agr"
                            elif (
                                annotation[lemma_token] > 0
                                and previous_token == "B-Agr"
                                and annotated_sentence[ix] == "O"
                            ):
                                annotated_sentence[ix] = "I-Agr"
                                previous_token = "I-Agr"
                elif len(annotation) >= 2:
                    for ix in range(len(sentence.split(" ")) - len(annotation)):
                        lemma_tokens = []
                        for ix_token in range(0, len(annotation)):
                            lemma_tokens.append(
                                get_lemma(sentence.split(" ")[ix_token + ix])
                            )
                        for ix_annotation, annotation_part in enumerate(
                            list(annotation.keys())
                        ):
                            if (
                                ix_annotation == 0
                                and lemma_tokens[ix_annotation] == annotation_part
                            ):
                                annotated_sentence[ix] = "B-Agr"
                            elif (
                                ix_annotation != 0
                                and lemma_tokens[ix_annotation] == annotation_part
                            ):
                                annotated_sentence[ix + ix_annotation] = "I-Agr"

                            # previous_token = "O"

    return annotated_sentence, list(set(found_concepts))


def annotate_all_abstracts(
    output_file: str,
    ontology_subset: List[str],
    excluded_terms: List[str],
    concepts_used_file_name: str,
    initial_abstract: int,
    last_abstract_to_annotate: int = 300000,
) -> None:
    """
    This function writes the annotated sentences based on the ontology subset
    with the IOB format in the output file. The sentences with excluded terms are not included.
    """
    print("[INFO] Lemmatizing excluded terms")
    excluded_terms_lemma = [
        get_lemma(excluded_term).strip()
        for excluded_term in excluded_terms
        if not " " in excluded_term and not "-" in excluded_term
    ]
    excluded_terms_lemma += [
        excluded_term.strip()
        for excluded_term in excluded_terms
        if " " in excluded_term or "-" in excluded_term
    ]
    all_found_concepts = []
    with open(output_file, "w", encoding="UTF8") as annotations_output_file:
        writer = csv.writer(annotations_output_file)
        writer.writerow(["text", "labels"])
        with open(ABSTRACTS_INPUT_FILE, "r") as abstracts_file:
            for ix, abstract in enumerate(abstracts_file):
                if ix > initial_abstract and ix < last_abstract_to_annotate:
                    print(f"Annotating abstract number: {ix}")
                    try:
                        if (
                            detect(abstract) == "en"
                        ):  # Checking the abstract is in English
                            print(f"{abstract[:20]} ...")
                            sentences = abstract.split(".")  # Dividing into sentences
                            for sentence in sentences:
                                if (
                                    len(sentence.strip().split(" "))
                                    > MIN_SENTENCE_WORDS
                                ):
                                    fixed_sentence = remove_mixed_words_from_sentence(
                                        sentence.strip()
                                    )
                                    # Extracting number of grammar problems
                                    matches = tool.check(sentence.strip())
                                    # Checking that the sentence is correct and starts with upper case.
                                    if (
                                        len(matches) <= MAX_GRAMMAR_WARNINGS
                                        and sentence.strip()[0].isupper()
                                        and sentence.count("(") == sentence.count(")")
                                        and (
                                            len(sentence.split(" ")[-1]) > 1
                                            or sentence.split(" ")[-1].isnumeric()
                                        )
                                    ):
                                        sentence = sentence.replace(
                                            ",", ""
                                        )  # Removing commas
                                        if not contains_excluded_term(
                                            sentence, excluded_terms_lemma
                                        ):
                                            # Getting annotations
                                            (
                                                sentence_annotations,
                                                found_concepts,
                                            ) = annotate_sentence(
                                                sentence.strip(), ontology_subset
                                            )

                                            all_found_concepts += found_concepts
                                            all_found_concepts = list(
                                                set(all_found_concepts)
                                            )
                                            # Only sentences that contain 1
                                            if "B-Agr" in sentence_annotations:
                                                writer.writerow(
                                                    [
                                                        sentence.strip(),
                                                        " ".join(sentence_annotations),
                                                    ]
                                                )
                                                annotations_output_file.flush()
                                            else:
                                                with open(
                                                    "non_annotated_sentences.txt",
                                                    "a",
                                                    encoding="UTF8",
                                                ) as non_annotated_sentences_file:
                                                    try:
                                                        non_annotated_sentences_file.write(
                                                            sentence.strip()
                                                        )
                                                        non_annotated_sentences_file.write(
                                                            "\n"
                                                        )
                                                        non_annotated_sentences_file.flush()
                                                    except (
                                                        Exception
                                                    ) as e_sentence_not_used:
                                                        print(e_sentence_not_used)
                        if ix % 1000 == 0:
                            print(all_found_concepts)
                            with open(
                                concepts_used_file_name, "w", encoding="UTF8"
                            ) as concepts_used_file:
                                for used_concept in all_found_concepts:
                                    try:
                                        concepts_used_file.write(used_concept.strip())
                                        concepts_used_file.write("\n")
                                        concepts_used_file.flush()
                                    except Exception as e_concepts_used:
                                        print(e_concepts_used)
                    except Exception as e:
                        print(e)
