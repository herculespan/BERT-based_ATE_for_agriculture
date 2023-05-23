# Agri-NER Experiments
(Full Documentation currently under development)

This repo contains the source code used for the paper **"Automatic extraction of agricultural terms from English text corpora using BERT"**.

The repo is self-contained. The notebook can be executed just by installing the necessary dependencies.

## Files
1. Dependencies: "*requeriments.txt"*.
2. The main file is the notebook "*Agri-NER-Experiments.ipynb"*. 
3. The *datasets* folder contains annotated sentences:
	a. Train-Val-Test Evaluation (Pref-Labels): *pref-agri-ner-input.csv*
	b. Synonyms Labels: *synonym-agri-ner-input.csv*
	c. Novel Labels: *novel-agri-ner-input.csv*
	d. Additional minified files used for speeding up the experiments and obtaining fast exploratory results.
4. The *labels* folder contains the label used for annotation.
5. The *utils* folder is a Python module that contains several libraries imported from *Agri-NER-Experiments.ipynb"*
6. The *data_structures* folder is a Python module that contains the API for working with the models.
7. The *experiment-analysis.sql* file contains some samples for SQL analysis on the results files.