"""

To run it:
python -m train

"""
import os
import traceback

from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
from eda_and_preprocessing import get_image_size, plot_class_distribution, do_content_analysis
from preprocessor import Preprocessor
from model import CNN



# Method to do preprocessing
def do_eda_and_preprocessing(dataset_path):
    # Get the size of an image for the input size
    img_path = os.path.join(dataset_path, "train/Apple___Apple_scab/0a14783a-838a-4d4f-a671-ff98011714c6___FREC_Scab 3288_new30degFlipLR.JPG")
    
    print("\nEDA: Image content analysis is starting!\n")
    # Get the size of an image
    width, height = get_image_size(img_path)
    
    # Plot the class distribution and save it under plots folder
    plot_class_distribution(dataset_path)

    # Plot all the data augmentation explorations
    do_content_analysis(img_path, height, width)

    print("\nEDA: Image content analysis ended!\n")
    print("\nImage Preprocessing is Starting!\n")
    try:
        # Do all the preprocessing and data augmentation 
        data_context = Preprocessor(width, height, dataset_path)
        print("\nImage Preprocessing Finised Successfully!\n")
        return data_context
    except Exception as e:
        print("\nError in data preprocessing!\n")
        print("Exception occurred:\n", e)
        print("Here's the full traceback:")
        traceback.print_exc()
        return


# Method to load and start model training
def start_training(data_context):
    print("\nModel Creation is Starting!\n")
    try:
        model_obj = CNN(data_context)

        # Perform hyperparameter tuning
        model_tuner = RandomSearch(
            model_obj.build_model,
            # build_model,
            objective='val_accuracy',
            max_trials = 3,
            executions_per_trial = 1,
            directory = 'output',
            project_name = 'PhytoPhinder_The_Leaf_Disease_Detective'
        ) 
        print("\nModel Loaded!\n")
        model_tuner.search_space_summary()
        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)

        print("\nModel Train is Starting!\n")
        model_tuner.search(
            data_context.train_generator,
            epochs = 10,
            validation_data = data_context.validation_generator,
            callbacks = [early_stopping]
        )
        # Get the optimal hyperparameters
        best_hps = model_tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('dense_1_units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)

        # Build the model with the optimal hyperparameters
        model = model_tuner.hypermodel.build(best_hps)
        # Retrain the model
        history = model.fit(data_context.train_generator, epochs = 10, validation_data = data_context.validation_generator)

        # Save the model
        model.save('models/best_model.h5')

    except Exception as e:
        print("\nERROR: Model Creation!\n")
        print("Exception occurred:\n", e)
        print("Here's the full traceback:")
        traceback.print_exc()
    


# The main method
def main() -> None:
    # Define the path to your dataset
    dataset_directory = 'small_dataset'
    
    # Method to do preprocessing
    data_context = do_eda_and_preprocessing(dataset_directory)
    if data_context:
        start_training(data_context)
        print("\nExperiment Finished!\n") 
    else:
        print("\nERROR: Experiment Finished!\n") 
    return



if __name__ == '__main__':
    main()