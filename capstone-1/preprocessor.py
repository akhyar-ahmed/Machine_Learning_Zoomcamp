
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Preprocessor class
class Preprocessor():
    def __init__(self, width, height, dataset_path):
        self.width = width
        self.height = height
        self.dataset_path = dataset_path
        self.seed = 42
        self.batch_size = 32
        self.train_datagen , self.test_datagen = self.get_image_data_gen()
        self.train_generator, self.validation_generator = self.create_train_test_generator()
        self.num_classes = self.get_num_classes()

    
    # Return image data generation with augmentation
    def get_image_data_gen(self):
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = "nearest"
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        return train_datagen, test_datagen
    

    # Return the training set and validation sets
    def create_train_test_generator(self):
        train_generator = self.train_datagen.flow_from_directory(
            os.path.join(self.dataset_path,"train"),
            target_size = (self.height, self.width),
            batch_size = self.batch_size,
            class_mode = 'categorical',
            shuffle = True,
            seed = self.seed
        )

        validation_generator = self.test_datagen.flow_from_directory(
            os.path.join(self.dataset_path,"test"),
            target_size = (self.height, self.width),
            batch_size = self.batch_size,
            class_mode = 'categorical',
            shuffle = True,
            seed = self.seed
        )
        return train_generator, validation_generator


    # Return number of classes
    def get_num_classes(self) -> int:
        num_classes = len(self.train_generator.class_indices)
        return num_classes