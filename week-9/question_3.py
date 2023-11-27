"""
Let's download and resize this image:

https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg

Based on the previous homework, what should be the target size for the image? (150, 150)

Now we need to turn the image into numpy array and pre-process it.
"""


import tensorflow.lite as tflite
import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image
from keras_image_helper import create_preprocessor


# method to download a image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


# method to resize image into target size
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


# method to preprocess the image tensor
def preprocess_input(X):
    X /= 127.5
    X -= 1.
    return X


def get_preprocessed_image(url, target_size=(150, 150)):    
    # load the image
    img = download_image(url)
    
    # resize the image
    img = prepare_image(img, target_size) # target_size = (150, 150)

    # preprocess image
    X = np.array(img, dtype='float32')
    X = np.array([X])
    X = preprocess_input(X)
    return X


# fastest way to do that using keras-image-helper package. (we need to install keras-image-helper)
# pipenv install keras-image-helper
def get_preprocessed_image_fast(url, target_size=(150, 150), model_name="xception"):
    preprocessor = create_preprocessor(model_name, target_size=target_size)
    X = preprocessor.from_url(url)
    return X


if __name__ == '__main__':
    url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"
    
    # get preprocessed image in a usual way!
    X = get_preprocessed_image(url, (150, 150))
    
    ## get preprocessed image in a faster way!
    # X = get_preprocessed_image_fast(url, (150, 150), "xception")

    # Assuming X is already prepared using the provided code
    # Access the value in the first pixel of the R channel
    r_channel_value = X[0, 0, 0, 0]

    print(f"Value in the first pixel of the R channel: {r_channel_value}")


"""
After the pre-processing, what's the value in the first pixel, the R channel?

0.3450980
0.5450980
0.7450980
0.9450980

Answer: ~0.9450980    
"""