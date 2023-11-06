# Model Training of Twitter Sentiment Analysis Project
----

This README provides instructions on how to train various models using the provided training script. To initiate the training process, run the following command:
```bash
python -m train
```

The training script allows flexibility in its usage through command-line arguments. Here's an overview of the available arguments and their descriptions:

## Command-Line Arguments:
* **--dataset_path (default: "asset", type: str)**: Path to the dataset folder. The training script will use data from this location.

* **--log_folder (type: str, default: "logs")**: Folder where log files will be saved during the training process. Log files contain training progress information.

* **--logging_steps (type: int, default: 100)**: Number of steps between logging training progress. For example, if set to 100, training progress will be logged every 100 steps.

* **--save_models (action: "store_true")**: If provided, trained models will be saved after training.

* **--model_name (type: str, default: "all", choices: ["all", "rfc", "lr", "xgb", "mlp"])**: Specifies the model to train. Options include "all" (train all available models), "rfc" (Random Forest Classifier), "lr" (Logistic Regression), and "xgb" (XGBoost Classifier). The default is the "all" which uses all the three models.
* **--seed (type: int, default: 42)**: Seed value for random number generation. Setting a seed ensures reproducibility of the results.

* **--batch_size (type: int, default: 32)**: Batch size for training data. It determines the number of samples used in each iteration of training.

* **--experiment_name (type: str, default: "")**: Name of the experiment. This is useful for organizing and identifying different experiments.

* **--experiment (type: str, default: "single_experiment", choices: ["single_experiment", "full_experiment"])**: Type of experiment to conduct. Options include "single_experiment" (a single training run) and "full_experiment" (a comprehensive experiment with hyperparameter tuning).

* **--search_hyperparameter (type: str, default: "yes", choices: ["yes", "no"])**: Specifies whether to perform hyperparameter search. If set to "yes," the script will perform a hyperparameter search to optimize the model's performance.

## Example Usage:
1. Train Random Forest Classifier on GPU, Save Models, and Perform Hyperparameter Search:

```bash
python -m train --experiment single_experiemnt --model_name rfc --save_models --search_hyperparameter yes
```

2. Train All models without Saving models and not Perform Hyperparameter search:
```bash
python -m train --experiment full_experiment --model_name all --search_hyperparameter no
```

3. Train Logistic Regression Model, Save Model and Train with a Batch Size of 32:
```bash
python -m train --model_name lr --save_models --batch_size 32
```


## Feel free to adjust the command-line arguments based on your requirements and experiment needs. Happy training!