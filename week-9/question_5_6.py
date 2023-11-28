"""
Now you need to copy all the code into a separate python file.
For the next two questions, we'll use a Docker image that we already prepared. This is the Dockerfile that we 
used for creating the image:

FROM public.ecr.aws/lambda/python:3.10
COPY bees-wasps-v2.tflite .

And pushed it to https://hub.docker.com/r/agrigorev/zoomcamp-bees-wasps/tags

A few notes:

    The image already contains a model and it's not the same model as the one we used for questions 1-4.
    The version of Python is 3.10, so you need to use the right wheel for TF-Lite. For Tensorflow 2.14.0, 
    it's https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

Question 5

Download the base image agrigorev/zoomcamp-bees-wasps:v2. You can easily make it by using docker pull command.


So what's the size of this base image?

    162 Mb
    362 Mb
    662 Mb
    962 Mb

You can get this information when running docker images - it'll be in the "SIZE" column.

Answer: 662Mb 

Now let's extend this docker image, install all the required libraries and add the code for lambda.

You don't need to include the model in the image. It's already included. The name of the file with the 
model is bees-wasps-v2.tflite and it's in the current workdir in the image (see the Dockerfile above for the reference). 
The provided model requires the same preprocessing for images regarding target size and rescaling the value range than used in homework 8.

Now run the container locally.

Score this image: https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg

To run it:
    python question_5_6.py
"""
from question_4 import get_predictions
from fastapi import FastAPI


app = FastAPI()


# translator directory
@app.get("/predict")
def get_prediction(image_url: str, model_path: str):
    img_url = image_url
    target_size = (150, 150)
    model_path = model_path

    prediction = get_predictions(img_url=img_url, model_path=model_path,\
                                  target_size=target_size, do_rescale_factor=False)
     
    prediction_name = None
    round_prediction = round(prediction[0][0], 2)
    if  round_prediction >= 0.5:
        prediction_name = "Bee"
    else:
        prediction_name = "Wasps"

    return {
        "Predicted_animal": prediction_name,
        "Predictions": str(round_prediction)
    }



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8001)