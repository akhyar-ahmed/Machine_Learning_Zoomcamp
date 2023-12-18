"""

To run it:
uvicorn app:app --host=0.0.0.0 --port=8001

"""
import json
import numpy as np
import gradio as gr

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI



# Open the JSON file
with open('./class_labels.json', 'r') as f:
    data = json.load(f)

# Now 'classes' are in a dictionary
classes = list(data.keys())


# Load the best model
best_model = load_model('./models/best_model.h5')


# Create a FastAPI application
app = FastAPI()

CUSTOM_PATH = "/predict"


# Function to predict sentiment from the tweet
async def predict_disease(uploaded_image):
    # Check if the image size is not 256x256
    if (uploaded_image.height,uploaded_image.width) != (256, 256):
        uploaded_image = uploaded_image.resize((256, 256))
    img_array = np.array(uploaded_image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get the predictions
    predictions = best_model.predict(img_array)
    
    # Assuming it's a multiclass classification, you can get the class with the highest probability
    predicted_class = np.argmax(predictions)
    return classes[predicted_class]

# Define the Gradio interface
image_input = gr.Image(type="pil", label="Upload your image here...")
prediction_output = gr.Label()


# Interface route for Gradio
interface = gr.Interface(fn=predict_disease, inputs=image_input, outputs=prediction_output, title="PhytoPhinder: The Leaf Disease Detective",  allow_flagging="never")
app = gr.mount_gradio_app(app, interface, path=CUSTOM_PATH)

