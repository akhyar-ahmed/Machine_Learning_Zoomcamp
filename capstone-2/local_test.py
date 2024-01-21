"""

To run it:
python local_test.py

"""
import json
import pandas as pd
import numpy as np
import torch

from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification
)
from configuration import Config
from preprocessor import load_dataset


# Get configuration object
config = Config()

config.deplpoyment = True

# Open the JSON file
with open('class_labels.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

# Now 'classes' are in a dictionary
classes = list(data.values())

sample_inputs = {
    "Class Index": [
        3, 4, 1, 2
    ],
    "Title": [
        "Fears for T N pension after talks", 
        "Group to Propose New High-Speed Wireless Format",
        "Man Sought  #36;50M From McGreevey, Aides Say (AP)",
        "Johnson Helps D-Backs End Nine-Game Slide (AP)"
    ],
    "Description": [
        "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.",
        " LOS ANGELES (Reuters) - A group of technology companies  including Texas Instruments Inc. &lt;TXN.N&gt;, STMicroelectronics  &lt;STM.PA&gt; and Broadcom Corp. &lt;BRCM.O&gt;, on Thursday said they  will propose a new wireless networking standard up to 10 times  the speed of the current generation.",
        "AP - The man who claims Gov. James E. McGreevey sexually harassed him was pushing for a cash settlement of up to  #36;50 million before the governor decided to announce that he was gay and had an extramarital affair, sources told The Associated Press.",
        "AP - Randy Johnson took a four-hitter into the ninth inning to help the Arizona Diamondbacks end a nine-game losing streak Sunday, beating Steve Trachsel and the New York Mets 2-0."
    ]
}

# Convert the dictionary to a Test DataFrame
test_df = pd.DataFrame(sample_inputs)

# Preprocess and get the tokenized data
test_dataloader = load_dataset(config, test_df)

# Load the quantized model
config_distilbert = DistilBertConfig.from_pretrained(
    config.pre_model,
    num_labels=config.num_classes,
)
best_model = DistilBertForSequenceClassification(config_distilbert)


# Load the trained model state_dict
if str(config.device) == "cpu":
    best_model.load_state_dict(torch.load("models/best_model_DistilBertForSequenceClassification_1e-05.bin", map_location=torch.device('cpu')))
else:
    best_model.load_state_dict(torch.load("models/best_model_DistilBertForSequenceClassification_1e-05.bin"))


best_model = best_model.to(config.device)

best_model.eval()

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

print()
for idx, row in test_df.iterrows():
    print(f"Title: {row['Title']} \nDescription: {row['Description']} \nTrue Class: {classes[y_true[idx]]} \nPredicted Class: {classes[y_pred[idx]]} ")
    print()
    print()
