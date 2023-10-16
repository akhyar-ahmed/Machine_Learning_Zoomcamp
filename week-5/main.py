"""

To run it:
waitress-serve --host=0.0.0.0 --port=8001 main:app


"""


import traceback
import sys
import os 

from flask import Flask, request, jsonify
from model import Model
from utils import predict


app = Flask(__name__)

# denote model and vectorizer path
if os.path.exists("assets/"):
    model_path = "assets/model1.bin"
    dv_path = "assets/dv.bin"
else:
    model_path = "./model2.bin"
    dv_path = "./dv.bin"

# Load the model and vectorizer object
model_obj = Model(model_path, dv_path)


# root directory
@app.route("/", methods=["GET"])
def root():
    return "Hello, Machine Learning Zoomcamp Homework for Week-5!"


# test a client
@app.route("/predict", methods=["POST"])
def get_client_probability():
    client = request.get_json()  
    try:
        # Q.3 get probability of getting credit to that particular client
        client_score = predict(model_obj, client)
        result = {
            "client score": client_score
        }
        return jsonify(result)

    except Exception as e:
        # Print the exception using sys.exc_info() method
        print("\nException caught using sys.exc_info():")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Exception Type:", exc_type)
        print("Exception Value:", exc_value)
        print("Exception Traceback:")
        return traceback.print_tb(exc_traceback)
        


# If it is, it calls the main() function
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)





