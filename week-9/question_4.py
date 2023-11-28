"""
Lets do a prediction for a single image.
image url: "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"

To run it:
    python question_4.py
"""
from question_2 import get_interpreter_input_output
from question_3 import get_preprocessed_image_fast, get_preprocessed_image


# get prediction from a pre_trained model
def get_predictions(img_url, model_path, target_size, model_name="xception", do_rescale_factor=False):
    # preprocess the input
    # X = get_preprocessed_image_fast(img_url, target_size, model_name)
    X = get_preprocessed_image(img_url, target_size, do_rescale_factor)

    # load the model
    interpreter, input_indx, output_indx = get_interpreter_input_output(model_path)
    
    # invoke inputs
    interpreter.set_tensor(input_indx, X)
    interpreter.invoke()
    
    # do predictions for the given image
    predictions = interpreter.get_tensor(output_indx)
    return predictions


if __name__ == '__main__':
    img_url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"
    target_size = (150, 150)
    model_path = "./models/bees-wasps.tflite"
    model_name = "xception"
    prediction = get_predictions(img_url, model_path, target_size, model_name, False)

    print(f"Prediction: {prediction[0]}")


"""
Now let's apply this model to this image. What's the output of the model?

    0.258
    0.458
    0.658
    0.858

Answer: 0.258
"""