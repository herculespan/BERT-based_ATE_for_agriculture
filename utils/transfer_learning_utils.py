def freeeze_embeddings(model, model_name: str) -> None:
    """It freezes all mndel's layers"""
    if model_name == "roberta-base":
        for param in model.roberta.roberta.embeddings.parameters():
            param.requires_grad = False
    elif model_name == "xlnet-base-cased":
        for param in model.xlnet.transformer.word_embedding.parameters():
            param.requires_grad = False
    else:  # BERT-based
        for param in model.bert.bert.embeddings.parameters():
            param.requires_grad = False


def freeeze_layers(model, num_frozen_layers: int, model_name: str) -> None:
    """It freezes all model's layers"""
    if model_name == "roberta-base":
        for layer in model.roberta.roberta.encoder.layer[:num_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    else:  # BERT-based
        for layer in model.bert.bert.encoder.layer[:num_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False


def freeeze_all_layers(model, model_name: str) -> None:
    """It freezes all mnodel's layers"""
    if model_name == "roberta-base":
        for param in model.roberta.roberta.parameters():
            param.requires_grad = False
    else:  # BERT-based
        for param in model.bert.bert.parameters():
            param.requires_grad = False
