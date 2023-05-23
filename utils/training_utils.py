import torch
from torch.utils.data import DataLoader

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import copy
from tqdm import tqdm

from utils.sequence_utils import DataSequence

from config import BATCH_SIZE
from config import EPOCHS
from config import PATIENCE
from config import IMPROVEMENT_RATIO


def train_loop(
    model, df_train, df_val, tokenizer, labels_to_ids, optimizer_name, learning_rate
):
    early_stopping_counter = 0
    train_dataset = DataSequence(df_train, tokenizer, labels_to_ids)
    val_dataset = DataSequence(df_val, tokenizer, labels_to_ids)

    train_dataloader = DataLoader(
        train_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=PATIENCE)

    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):
            train_label = train_label[0].to(device)
            mask = train_data["attention_mask"][0].to(device)
            input_id = train_data["input_ids"][0].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]

            predictions = logits_clean.argmax(dim=1)

            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0

            for val_data, val_label in val_dataloader:
                val_label = val_label[0].to(device)
                mask = val_data["attention_mask"][0].to(device)

                input_id = val_data["input_ids"][0].to(device)

                loss, logits = model(input_id, mask, val_label)

                logits_clean = logits[0][val_label != -100]
                label_clean = val_label[val_label != -100]

                predictions = logits_clean.argmax(dim=1)

                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

            train_loss = total_loss_train / len(df_train)
            train_accuracy = total_acc_train / len(df_train)
            val_accuracy = total_acc_val / len(df_val)
            val_loss = total_loss_val / len(df_val)
            if val_accuracy > best_acc * IMPROVEMENT_RATIO:
                print(f"[INFO] Updating weights: {val_accuracy} > {best_acc}")
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = val_accuracy
                best_epoch = epoch_num
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print(
                f"Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}"
            )

            if early_stopping_counter == PATIENCE:
                print(
                    f"[INFO] {early_stopping_counter} patience epochs reached in epoch {epoch_num}"
                )
                model.load_state_dict(best_model_wts)
                return train_loss, train_accuracy, val_loss, val_accuracy, best_epoch

            scheduler.step(val_loss)
    # Loading the weights that obtained the best Val. Accuracy
    model.load_state_dict(best_model_wts)

    return train_loss, train_accuracy, val_loss, val_accuracy, best_epoch
