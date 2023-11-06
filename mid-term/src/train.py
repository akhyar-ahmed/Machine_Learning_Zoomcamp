"""

To run it:
python -m train

"""
import logging
import time
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from dataloader import load_dataset
from utils import init_experiments, read_args, write_results, get_model_name
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from model import do_hyperparameter_tuning, get_model


warnings.filterwarnings("ignore")



# Train the model
def train(features, labels, model):
    model.fit(features, labels)
    return model


# log_dir is the projects log_dir
# Extension dir is the last part of the output directory, e.g. "saved_model_and_results"
def eval(args, model_name, data_context, best_model, log_dir, extension_dir, test_on_valid=False) -> list:
    output_dir = os.path.join(log_dir, extension_dir)
    # Check if the folder exists
    if not os.path.exists(output_dir):
        # If not, create the folder
        os.makedirs(output_dir)
        logging.info(f"Folder '{output_dir}' created successfully.")
    else:
        logging.info(f"Folder '{output_dir}' already exists.")

    # Initialize empty lists to store predictions and true labels
    predictions = []
    true_labels = []
    
    # Decide the test dataloader
    test_dataloader = data_context.valid_dataloader if test_on_valid else data_context.test_dataloader
    if test_on_valid:
        logging.info("Evaluation is starting on the Valid dataset!!")
    else:
        logging.info("Evaluation is starting on the Test dataset!!")
    # Iterate through the validation DataLoader
    for batch in test_dataloader:
        features, labels = batch["text"], batch["label"]
        
        # Convert PyTorch tensors to numpy arrays
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        # Make predictions using the Random Forest Classifier
        preds = best_model.predict(features_np)
        
        # Extend the predictions and true labels lists
        predictions.extend(preds)
        true_labels.extend(labels_np)

    logging.info("Evaluation done!!")
    f1 = round(f1_score(true_labels, predictions, average = "weighted"), 2)
    # Display F1 Score
    logging.info(f"F1 score for {model_name}: {f1}")

    # Calculate accuracy
    accuracy = round(accuracy_score(true_labels, predictions), 2)
    # Display F1 Score
    logging.info(f"Accuracy score for {model_name}: {accuracy}")

    # Compute Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    # Display and Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {model_name}")
    # Save the confusion matrix figure
    plt.savefig(output_dir+f"/{model_name}_confusion_matrix.png")  
    
    # Save the best-performing model
    if args.save_models:
        joblib.dump(best_model, output_dir+f"/{model_name}_best_model.pkl")
        logging.info(f"Best model is saved in folder {log_dir}/{output_dir}+/best_model.pkl successfully.")
    return [accuracy, f1, output_dir]



# Single experiment method
def single_experiment(args, logs_dir, test_on_valid=False) -> None:
    experiment_start_time = time.time()
    model_name = None
    # Prepare datasets and dataloaders for the experiment
    data_context = load_dataset(args)
    logging.info("Dataset loaded!!")
    
    train_features, train_labels = [], []
    for batch in data_context.train_dataloader:
        batch_features, batch_labels = batch["text"], batch["label"]
        train_features.append(batch_features.numpy())
        train_labels.append(batch_labels.numpy())
    features = np.vstack(train_features)
    labels = np.concatenate(train_labels)
    
    assert len(features) == len(labels)
    logging.info("Features extracted for hyperparameter tuning!!")
    
    assert args.model_name != "all"
    model_name = get_model_name(args.model_name)
    if args.search_hyperparameter == "yes":
        logging.info(f"Experiment is starting to find best parameters for {model_name}!!")
        logging.info("Model training is starting!!")
        starttime = time.time()
        model = do_hyperparameter_tuning(features, labels, model_name)
        duration = (time.time()-starttime)
        logging.info(f"Training finished in {(duration/60):.2f} minutes")
    else:
        model = get_model(model_name)
        logging.info("Model training is starting!!")
        starttime = time.time()
        model = train(features, labels, model)
        duration = (time.time()-starttime)
        logging.info(f"Training finished in {(duration/60):.2f} minutes")

    # Evaluate the model and save the results.
    starttime = time.time()
    results = eval(args, model_name, data_context, model, logs_dir, "saved_model_and_results", test_on_valid)
    duration = (time.time()-starttime)

    # Write the results into CSV's
    write_results(results[2], f"accuracy", results[0], model_name)
    write_results(results[2], f"f1", results[1], model_name)
    write_results(results[2], f"predict_duration", round(duration, 2), model_name)
    
    duration = (time.time()-experiment_start_time)
    logging.info(f"Experiment is finished in {(duration/60):.2f} minutes")
    return


# Method to do full experiments
def full_experiment(args, log_dir):
    assert args.model_name == "all"
    logging.info(f"Full experiment is starting!!")
    for name in ["lr", "xgb", "rfc"]:
        args.model_name = name
        single_experiment(args, log_dir)
    return


# The main method
def main() -> None:
    try:
        starttime = time.time()
        args = read_args()
        if args.experiment == "single_experiment":
            name = f"..model={args.model_name}"
            log_dir = init_experiments(args, name + "..single_experiment")
            single_experiment(args, log_dir, True)
        elif args.experiment == "full_experiment":
            log_dir = init_experiments(args, "full_experiment")
            full_experiment(args, log_dir)

        duration = (time.time() - starttime) / 60
        logging.info(f"Complete experiment is finished in {duration:.2f} minutes")
    except Exception as e:
        logging.exception(e)
    return


if __name__ == '__main__':
    main()
