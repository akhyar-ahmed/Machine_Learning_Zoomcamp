"""

To run it:
uvicorn app:app --host=0.0.0.0 --port=8001

"""
import json
import os
import pandas as pd
import numpy as np
import torch
import gradio as gr

from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification
)
from configuration import Config
from preprocessor import load_dataset
from fastapi import FastAPI


# Get configuration object
config = Config()

config.deplpoyment = True

# Open the JSON file
with open('class_labels.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

# Now 'classes' are in a dictionary
classes = list(data.values())

# Load the quantized model
config_distilbert = DistilBertConfig.from_pretrained(
    config.pre_model,
    num_labels=config.num_classes,
)
best_model = DistilBertForSequenceClassification(config_distilbert)


# Load the trained model state_dict
if str(config.device) == "cpu":
    best_model.load_state_dict(torch.load("./models/best_model_DistilBertForSequenceClassification_1e-05.bin", map_location=torch.device('cpu')))
else:
    best_model.load_state_dict(torch.load("./models/best_model_DistilBertForSequenceClassification_1e-05.bin"))


best_model = best_model.to(config.device)

best_model.eval()


# Create a FastAPI application
app = FastAPI()

CUSTOM_PATH = "/predict"


# Function to predict sentiment from the tweet
async def predict_news(title, description):
    sample_inputs = {
        "Class Index": [1],
        "Title": [title],
        "Description": [description]
    }
    # Convert the dictionary to a Test DataFrame
    test_df = pd.DataFrame(sample_inputs)

    # Preprocess and get the tokenized data
    test_dataloader = load_dataset(config, test_df)

    y_pred = None
    y_true = None
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(config.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(config.device)
            labels = batch['label'].to(config.device)
            
            outputs = best_model(input_ids, attention_mask)
            y_true = labels.cpu().numpy()
            y_pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    
    return classes[y_pred[0]]


# Define the Gradio interface components
title_input = gr.Textbox(label="Title", placeholder="Enter title here...")
description_input = gr.Textbox(label="Description", placeholder="Enter description here...")


# Interface route for Gradio
interface = gr.Interface(
    fn=predict_news, 
    inputs=[title_input, description_input],
    outputs=[gr.Textbox(label="Predicted News Category")], 
    title="Trans4News: Multiclass News Classifier",  
    allow_flagging="never"
)
app = gr.mount_gradio_app(app, interface, path=CUSTOM_PATH)

