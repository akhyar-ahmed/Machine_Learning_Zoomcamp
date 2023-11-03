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






# If it is, it calls the main() function
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)