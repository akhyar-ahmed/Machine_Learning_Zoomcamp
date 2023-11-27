"""
To be able to use this model, we need to know the index of the input and the index of the output.

To Run it:
    python question_2.py
"""

import tensorflow.lite as tflite


def get_interpreter_input_output(model_path="./models/bees-wasps.tflite"):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()
    output_index = interpreter.get_output_details()
    
    print(f"Input Details:\n{input_index}\n")
    print(f"Output Details:\n{output_index}\n")
    
    return interpreter, input_index[0]['index'], output_index[0]['index']


if __name__ == '__main__':
    _, input, output = get_interpreter_input_output()
    print(f"Input index: {input}, Output index: {output}\n")


"""
What's the output index for this model?

    3
    7
    13
    24

Answer: 13
"""