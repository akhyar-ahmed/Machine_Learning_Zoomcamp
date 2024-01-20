"""

To run it:
python -m train

"""
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    DistilBertConfig,
    DistilBertForSequenceClassification
)
from configuration import Config
from preprocessor import load_dataset




# Set random seed for reproducibility
def set_seed(seed) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(">> SEEDING DONE")


# Method to load and start model training
def train(context, config_obj, writer) -> None:
    for model_name in [
        "DistilBertForSequenceClassification",
        "BertForSequenceClassification",
    ]:
        for lr in [1e-5, 1e-3]:
            # Model definition
            if model_name == "BertForSequenceClassification":
                # Bert configuration
                config_bert = BertConfig.from_pretrained(
                    config_obj.pre_model,
                    num_labels=config_obj.num_classes,
                )
                model = BertForSequenceClassification(config_bert)
            elif model_name == "DistilBertForSequenceClassification":
                # DistilBert configuration
                config_distilbert = DistilBertConfig.from_pretrained(
                    config_obj.pre_model,
                    num_labels=config_obj.num_classes,
                )
                model = DistilBertForSequenceClassification(config_distilbert)

            # Model to device
            model = model.to(config_obj.device)

            # Define optimizer and criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            best_val_loss = float("inf")
            early_stopping_counter = 0
            for epoch in range(config_obj.epochs):
                model.train()
                train_loss = 0.0
                for batch in tqdm(
                    context.train_dataloader, desc=f"Epoch {epoch + 1}/{config_obj.epochs}"
                ):
                    input_ids = batch["input_ids"].squeeze(1).to(config_obj.device)
                    attention_mask = (
                        batch["attention_mask"].squeeze(1).to(config_obj.device)
                    )
                    labels = batch["label"].to(config_obj.device)

                    optimizer.zero_grad()

                    if model_name == "DistilBertForSequenceClassification":
                        outputs = model(input_ids, attention_mask)
                    elif model_name == "BertForSequenceClassification":
                        token_type_ids = (
                            batch["token_type_ids"].squeeze(1).to(config_obj.device)
                        )
                        outputs = model(input_ids, attention_mask, token_type_ids)

                    loss = criterion(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0.0
                y_true = []
                y_pred = []

                with torch.no_grad():
                    for batch in tqdm(context.valid_dataloader, desc=f"Validation:"):
                        input_ids = batch["input_ids"].squeeze(1).to(config_obj.device)
                        attention_mask = (
                            batch["attention_mask"].squeeze(1).to(config_obj.device)
                        )
                        labels = batch["label"].to(config_obj.device)

                        if model_name == "DistilBertForSequenceClassification":
                            outputs = model(input_ids, attention_mask)
                        elif model_name == "BertForSequenceClassification":
                            token_type_ids = (
                                batch["token_type_ids"].squeeze(1).to(config_obj.device)
                            )
                            outputs = model(input_ids, attention_mask, token_type_ids)

                        loss = criterion(outputs.logits, labels)
                        val_loss += loss.item()

                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

                val_loss /= len(context.valid_dataloader)
                accuracy_val = accuracy_score(y_true, y_pred)
                f1_val = f1_score(y_true, y_pred, average="micro")

                print(
                    f"Epoch {epoch + 1}/{config_obj.epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {accuracy_val}, Validation F1: {f1_val}"
                )

                # Tensorboard logging
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Val", val_loss, epoch)
                writer.add_scalar("Accuracy/Val", accuracy_val, epoch)
                writer.add_scalar("F1/Val", f1_val, epoch)

                # Early stopping and model checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    torch.save(
                        model.state_dict(), f"models/best_model_{model_name}_{lr}.bin"
                    )  # Save the best model
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= config_obj.patience:
                    print("Early stopping. Training stopped.")
                    break
    return 


# Perform evaluation
def test(context, config_obj) -> None:
    # Instantiate the DistilBERT model with the number of classes in your dataset
    config_distilbert = DistilBertConfig.from_pretrained(
        config_obj.pre_model,
        num_labels=config_obj.num_classes,
    )
    best_model = DistilBertForSequenceClassification(config_distilbert)


    # Load the trained model state_dict
    best_model.load_state_dict(torch.load("models/best_model_DistilBertForSequenceClassification_1e-05.bin"))

    best_model = best_model.to(config_obj.device)
    # Evaluate the model on the test set
    best_model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(
                    context.test_dataloader, desc=f"Epoch {epoch + 1}/{config_obj.epochs}"
                ):
            input_ids = batch['input_ids'].squeeze(1).to(config_obj.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(config_obj.device)
            labels = batch['label'].to(config_obj.device)
            
            outputs = best_model(input_ids, attention_mask)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

    # Calculate and print metrics
    accuracy_test = accuracy_score(y_true, y_pred)
    f1_test = f1_score(y_true, y_pred, average='micro')
    classification_report_test = classification_report(y_true, y_pred)

    print(f"Test Accuracy: {accuracy_test}")
    print(f"Test F1 Score: {f1_test}")
    print("Classification Report:\n", classification_report_test)
    return



# Start the training process
def start_experiment(config_obj) -> None:
    # Preprocess and create dataloader
    context = load_dataset(config_obj)

    # Instantiate a Tensorboard SummaryWriter for logging
    writer = SummaryWriter()

    # Train the model with hyperparameter tuning
    # train(context, config_obj, writer)

    # Evaluate the best model using test set
    test(context, config_obj)

    # Close the Tensorboard SummaryWriter
    writer.close()
    return


# The main method
def main() -> None:
    # Load all the configurations
    config_obj = Config()

    # Set a random seed
    set_seed(Config().seed)

    # Start the training process
    start_experiment(config_obj)



if __name__ == '__main__':
    main()