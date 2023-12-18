import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class CNN():
    def __init__(self, data_context) -> None:
        self.data_context = data_context
    

    # Define the model with hyper-parameter combinations
    def build_model(self, hp):
        model = Sequential()
        model.add(Conv2D(filters = hp.Int('conv_1_filter', min_value = 32, max_value = 64, step = 32),
                        kernel_size = hp.Choice('conv_1_kernel', values = [3]),
                        activation = 'relu',
                        input_shape = (self.data_context.width, self.data_context.height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=hp.Int('conv_2_filter', min_value=64, max_value=96, step=32),
                    kernel_size=hp.Choice('conv_2_kernel', values = [3]),
                    activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=hp.Int('conv_3_filter', min_value=96, max_value=128, step=32),
                    kernel_size=hp.Choice('conv_3_kernel', values = [3]),
                    activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=hp.Int('dense_1_units', min_value=64, max_value=128, step=32),
                        activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.3, default=0.1, step=0.10)))
        model.add(Dense(self.data_context.num_classes, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[0.01, 0.001, 0.002])),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model