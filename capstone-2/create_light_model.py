""""
Convert a model from Keras to TF-Lite format.

To Run it:
python create_light_model.py
"""

import torch

from configuration import Config
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification
)

config_obj = Config()

# Instantiate the DistilBERT model with the number of classes in your dataset
config_distilbert = DistilBertConfig.from_pretrained(
    config_obj.pre_model,
    num_labels=config_obj.num_classes,
)
# Load original model
original_model = DistilBertForSequenceClassification(config_distilbert)

if str(config_obj.device) == "cpu":
    original_model.load_state_dict(torch.load("./models/best_model_DistilBertForSequenceClassification_1e-05.bin", map_location=torch.device('cpu')))
else:
    original_model.load_state_dict(torch.load("./models/best_model_DistilBertForSequenceClassification_1e-05.bin"))
original_model.eval()

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    original_model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "./models/quantized_best_model.pt")
print(">> Quantized Model Created!")
