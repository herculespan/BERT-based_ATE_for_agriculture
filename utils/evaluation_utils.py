import torch
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall
from torch.utils.data import DataLoader
from utils.sequence_utils import DataSequence
import numpy as np
from utils.preprocessing_utils import align_word_ids
from typing import List

from config import MAX_LENGTH, LABEL_ALL_TOKENS, BATCH_SIZE


def get_entities_positions(annotated_sentence: List[str]):
    """Auxiliar function used for computing the entity performance"""
    entities = []
    for ix, annotation in enumerate(annotated_sentence):
        if annotation.startswith("B"):
            try:
                if is_new_entity:
                    entities.append(entity_positions)
            except:
                pass
            entity_positions = set()
            is_new_entity = True
            entity_positions.add(ix)
            if ix == len(annotated_sentence) - 1:
                entities.append(entity_positions) 
        elif annotation.startswith("O"):
            is_new_entity = False
            try: 
                if len(entity_positions) > 0:
                    entities.append(entity_positions)
                    entity_positions = set()
            except:
                pass
        elif annotation.startswith("I") and is_new_entity:
            entity_positions.add(ix)

    return sorted(entities)

def get_entity_annotation_report(df_test, model, tokenizer, ids_to_labels):
    """
    It returns the performances per entity (not per label).
    """
    total_real_entities = 0
    total_complete_entities_annotated = 0
    total_partial_entities_annotated = 0
    total_false_positives = 0
    for sentence, annotation in zip(df_test['text'], df_test['labels']):
        predicted_annotations = get_sentence_predicted_annotation(model, tokenizer, sentence, ids_to_labels)
        real_annotations = annotation.split(" ")
        if len(predicted_annotations) == len(real_annotations):
            real_entities = get_entities_positions(real_annotations)
            predicted_entities = get_entities_positions(predicted_annotations)
            num_complete_entities_annotated, num_partial_entities_annotated, num_false_positive_annotations = get_entity_annotation_metrics(real_entities, predicted_entities)
            total_real_entities+=len(real_entities)
            total_complete_entities_annotated+=num_complete_entities_annotated
            total_partial_entities_annotated+=num_partial_entities_annotated
            total_false_positives+=num_false_positive_annotations
    
    return round(np.mean(total_complete_entities_annotated/total_real_entities), 4),\
           round(np.mean(total_partial_entities_annotated/total_real_entities), 4),\
           round(np.mean(total_false_positives/total_real_entities), 4)

def get_annotated_entities(sentence: str, annotations: List[str]):
    """
    It returns the entities of a sentence by using the IOB annotations.
    """
    annotated_entities = []
    entities = get_entities_positions(annotations)
    for entity in entities:
        if len(entity) == 1:
            annotated_entities.append(sentence.split(" ")[list(entity)[0]])
        elif len(entity) >= 2:
            annotated_entities.append(sentence.split(" ")[list(entity)[0]:list(entity)[-1]+1])
    return annotated_entities

def get_entity_annotation_metrics(real_annotations, predicted_annotations):
    num_complete_entity_annotated = 0
    num_partial_entity_annotated = 0
    num_false_positive_annotations = 0
    for predicted_annotation in predicted_annotations:
        is_true_positive = False
        for real_annotation in real_annotations:
            if real_annotation == real_annotation.intersection(predicted_annotation):
                is_true_positive = True
                num_complete_entity_annotated+=1
            elif len(real_annotation.intersection(predicted_annotation)) != 0:
                is_true_positive = True
                num_partial_entity_annotated+=1
        if not is_true_positive:
            num_false_positive_annotations+=1
    return num_complete_entity_annotated, num_partial_entity_annotated, num_false_positive_annotations

def get_sentence_predicted_annotation(model, tokenizer, sentence, ids_to_labels):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.to(device)

    text = tokenizer(
        sentence,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    mask = text["attention_mask"][0].unsqueeze(0).to(device)

    input_id = text["input_ids"][0].unsqueeze(0).to(device)
    label_ids = torch.Tensor(align_word_ids(sentence, tokenizer)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    return prediction_label

def evaluate(model, df_test, tokenizer, unique_labels):
    """It returns the performances per label"""
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    test_dataset = DataSequence(df_test, tokenizer, labels_to_ids)

    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0
    all_labels = []
    all_preds = []
    for test_data, test_label in test_dataloader:
        test_label = test_label[0].to(device)
        mask = test_data["attention_mask"][0].to(device)
        input_id = test_data["input_ids"][0].to(device)

        loss, logits = model(input_id, mask, test_label.long())

        logits_clean = logits[0][test_label != -100]
        label_clean = test_label[test_label != -100]
        all_labels += label_clean
        predictions = logits_clean.argmax(dim=1)
        all_preds += predictions

        acc = (predictions == label_clean).float().mean()
        total_acc_test += acc

    test_accuracy = total_acc_test / len(df_test)

    f1_macro = F1Score(task='multiclass', num_classes=3, average="macro")
    test_f1 = f1_macro(torch.tensor(all_preds), torch.tensor(all_labels))

    f1_all = F1Score(task='multiclass', num_classes=3, average=None)
    test_f1_all = f1_all(torch.tensor(all_preds), torch.tensor(all_labels))
    f1_per_class = {
        ids_to_labels[ix]: performance
        for ix, performance in enumerate(test_f1_all.tolist())
    }

    precision_macro = Precision(task='multiclass', num_classes=3, average="macro")
    test_precision = precision_macro(torch.tensor(all_preds), torch.tensor(all_labels))

    precision_all = Precision(task='multiclass', num_classes=3, average=None)
    test_precision_all = precision_all(
        torch.tensor(all_preds), torch.tensor(all_labels)
    )
    precision_per_class = {
        ids_to_labels[ix]: performance
        for ix, performance in enumerate(test_precision_all.tolist())
    }

    recall_macro = Recall(task='multiclass', num_classes=3, average="macro")
    test_recall = recall_macro(torch.tensor(all_preds), torch.tensor(all_labels))
    recall_all = Recall(task='multiclass', num_classes=3, average=None)
    test_recall_all = recall_all(torch.tensor(all_preds), torch.tensor(all_labels))
    recall_per_class = {
        ids_to_labels[ix]: performance
        for ix, performance in enumerate(test_recall_all.tolist())
    }

    print(f"Test Accuracy: {test_accuracy: .3f}")
    return (
        test_accuracy,
        test_f1,
        f1_per_class,
        test_precision,
        precision_per_class,
        test_recall,
        recall_per_class,
    )