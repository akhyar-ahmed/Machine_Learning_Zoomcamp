import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re

from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    BertTokenizer,
    DistilBertTokenizer,
)



# Save all infos about training data in one place
class DataContext:
    def __init__(self, config) -> None:
        self.bert_tokenizer = BertTokenizer.from_pretrained(config.pre_model)
        self.distil_bert_tokenizer = DistilBertTokenizer.from_pretrained(
            config.pre_distil_model
        )
        self.vectorizer = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.df_train = None
        self.df_test = None
        self.df = None

    # preprocessing method for all texts
    def preprocess_texts(self) -> None:
        preprocessed_texts_ls = []
        for ix, row in self.df.iterrows():
            text = row.text

            # Convert to lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

            # Remove mentions and hashtags
            text = re.sub(r"@\w+|\#", "", text)

            # Remove special characters and punctuation
            text = re.sub(r"[^\w\s]", "", text)

            # Remove spaces
            text = re.sub(r"\s+", " ", text)

            # Remove unnecessary dots
            text = re.sub(r"\.{2,}", ".", text)

            # Remove dots at the beginning or end of the sentence
            text = text.strip(".")

            # Remove spaces at the beginning or end of the sentence
            text = text.strip(" ")

            preprocessed_texts_ls.append(text)

        # Create new columns in our main df
        self.df["preprocessed_news"] = preprocessed_texts_ls
        return


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Subtract 1 from each value in the 'labels' column
        self.dataframe["labels"] = self.dataframe["labels"] - 1

    def __len__(self):
        return len(self.dataframe.labels)

    def __getitem__(self, idx):
        if "token_type_ids" in self.dataframe.columns.tolist():
            sample = {
                "input_ids": self.dataframe["input_ids"].iloc[idx],
                "attention_mask": self.dataframe["attention_mask"].iloc[idx],
                "token_type_ids": self.dataframe["token_type_ids"].iloc[idx],
                "label": torch.tensor(
                    self.dataframe["labels"].iloc[idx], dtype=torch.long
                ),
            }
        else:
            sample = {
                "input_ids": self.dataframe["input_ids"].iloc[idx],
                "attention_mask": self.dataframe["attention_mask"].iloc[idx],
                "label": torch.tensor(
                    self.dataframe["labels"].iloc[idx], dtype=torch.long
                ),
            }
        return sample


# Perform tookenization for every news text
def do_tokenization(config, context, model_name) -> BertTokenizer:
    # Tokenize each news in the DataFrame
    def tokenize_news(news_text):
        if model_name == "BertToken":
            tokens = context.bert_tokenizer(
                news_text,
                truncation=True,
                padding="max_length",
                max_length=config.max_length,
                return_tensors="pt",
            )
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                tokens[key] = torch.LongTensor((tokens[key]))
        elif model_name == "DistilBertToken":
            tokens = context.distil_bert_tokenizer(
                news_text,
                truncation=True,
                padding="max_length",
                max_length=config.max_length,
                return_tensors="pt",
            )
            for key in ["input_ids", "attention_mask"]:
                tokens[key] = torch.LongTensor((tokens[key]))
        return tokens

    tokenized_news = context.df["preprocessed_news"].apply(tokenize_news)
    return tokenized_news


# Dataset can be either train, test or valid
def load_dataset(config) -> DataContext:
    context = DataContext(config)

    # Read the dataset from config.train_PATH and config.test_PATH
    context.df_train = pd.read_csv(config.train_PATH, encoding="utf-8")

    context.df_test = pd.read_csv(config.test_PATH, encoding="utf-8")

    # Merge DataFrames
    context.df = pd.concat([context.df_train, context.df_test], ignore_index=True)

    # Assuming df is your DataFrame
    context.df["text"] = context.df["Title"] + " " + context.df["Description"]

    # Rename 'Class Index' to 'label'
    context.df = context.df.rename(columns={"Class Index": "labels"})

    # Print a log
    print(">> CSV LOADING DONE.")

    # Preprocess and create encodings for the dataset
    context.preprocess_texts()

    # Print a log
    print(">> DATA PREPROCESSING DONE.")

    # Tokenize each news and add the 'input_ids', 'attention_mask', and 'token_type_ids' columns
    tokenized_news = do_tokenization(
        config, context, "DistilBertToken"
    )  # you can change it here for BertToken
    context.df = pd.concat([context.df, pd.DataFrame(tokenized_news.tolist())], axis=1)

    # Print a log
    print(">> DATA TOKENIZATION DONE.")

    # Create a custom dataset instance
    dataset = CustomDataset(context.df)

    # Define the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # Create context dataset for hyperparameter tuning
    context.train_dataset = train_dataset
    context.valid_dataset = val_dataset
    context.test_dataset = test_dataset

    # Create data loaders for train, validation, and test sets
    context.train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
    )
    context.valid_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    context.test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Print a log
    print(">> DATALOADER AND VALIDATION FRAMEWORK CREATED.")

    # Show max-length for each category
    showLengthCount(context)

    return context


# Show maximum length for each class news
def showLengthCount(context) -> None:
    df = context.df.copy()
    for i, row in enumerate(df.iterrows()):
        df.loc[i, "length"] = len(df.loc[i, "preprocessed_news"].split())
    print(">> Count of news : ", len(df))
    labels = list(df.labels.unique())
    for label in labels:
        df_label = df[df["labels"] == label]
        print(f">> Max length for class: {label} is : {df_label.length.unique().max()}")
    return