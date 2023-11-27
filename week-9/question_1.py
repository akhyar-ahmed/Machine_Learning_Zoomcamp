""""
Convert a model(download model from: https://github.com/alexeygrigorev/large-datasets/releases/download/wasps-bees/bees-wasps.h5)\
    from Keras to TF-Lite format.

To Run it:
python question_1.py
"""


import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "./models/bees-wasps.h5"

# Load the saved Keras model
loaded_model = load_model(model_path)

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('./models/bees-wasps.tflite', 'wb') as f:
    f.write(tflite_model)


# Print the size of the converted model
converted_model_size = len(tflite_model) / (1024 * 1024)  # Size in megabytes
print(f"Size of the converted model: {converted_model_size:.2f} MB")

"""
What's the size of the converted model?

    21 Mb
    43 Mb
    80 Mb
    164 Mb

Answer: ~43 Mb
"""