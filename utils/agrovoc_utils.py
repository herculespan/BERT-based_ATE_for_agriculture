from config import INPUT_DATA_PATH
from config import OUTPUT_DATA_PATH


def load_agrovoc_prefLabels():
    from config import PREF_LABELS_FILE

    concepts = []
    with open(INPUT_DATA_PATH + PREF_LABELS_FILE, "r") as concepts_file:
        lines = concepts_file.readlines()
        concepts = [concept.strip() for concept in lines]
    return concepts


def load_agrovoc_synoyms_altLabels():
    from config import ALT_LABELS_TO_USE_AS_SYNONYMS

    concepts = []
    with open(OUTPUT_DATA_PATH + ALT_LABELS_TO_USE_AS_SYNONYMS, "r") as concepts_file:
        lines = concepts_file.readlines()
        concepts = [concept.strip() for concept in lines]
    return concepts


def load_agrovoc_used_prefLabels():
    from config import USED_PREF_LABELS_FILE

    concepts = []
    with open(OUTPUT_DATA_PATH + USED_PREF_LABELS_FILE, "r") as concepts_file:
        lines = concepts_file.readlines()
        concepts = [concept.strip() for concept in lines]
    return concepts


def load_agrovoc_altLabels(remove_bigrams: bool = False):
    from config import ALT_LABELS_FILE

    concepts = []
    with open(INPUT_DATA_PATH + ALT_LABELS_FILE, "r") as concepts_file:
        lines = concepts_file.readlines()
        if remove_bigrams:
            concepts = [concept.strip() for concept in lines if not " " in concept]
        else:
            concepts = [concept.strip() for concept in lines]
    return concepts
