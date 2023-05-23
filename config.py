# Filesystem Constants
INPUT_DATA_PATH = 'labels/'
PREF_LABELS_FILE = 'prefLabels.txt'
ALT_LABELS_FILE = 'cleanAltLabels.txt'
USED_PREF_LABELS_FILE = 'pref-labels-used.txt'
OUTPUT_DATA_PATH = 'output_data/'
BEST_MODELS_PATH = './best_models'
DATASETS_FOLDER = './datasets'
ABSTRACTS_INPUT_FILE = INPUT_DATA_PATH + 'all_abstracts.txt'

# Pre-processing constants
LABEL_ALL_TOKENS = True
MAX_LENGTH = 100

# Training Constants
BATCH_SIZE=1
EPOCHS=1
PATIENCE=10
NUM_EXPERIMENTS=5
IMPROVEMENT_RATIO = 1.0025

# Corpus Constants
MAX_GRAMMAR_WARNINGS = 1
MIN_SENTENCE_WORDS = 4
MIN_CONCEPT_LENGTH = 3
FINAL_PREF_ABSTRACT = 10000