"""

To run it:
python -m train

"""
import logging
import time

from dataloader import load_dataset
from utils import init_experiments, read_args, write_results
from sklearn.metrics import accuracy_score, f1_score, classification_report


# perform a hyperparameter search for optimal learning rate and batch size on hwu64 dataset for models bert and bert+adapter
def search_hyperparameter(args, log_dir):
    args.add_adapter=False
    for batch_size in [8,16,32,64,128,256]:
        for learning_rate in [1e-5, 3e-5, 5e-5]:
            args.non_adapter_train_batch_size=batch_size
            args.non_adapter_learning_rate=learning_rate
            log_prefix=f"model=bert,learning_rate={learning_rate},batch_size={batch_size},"
            single_experiment(args, log_dir, log_praefix=log_prefix, test_on_valid=True)

    args.add_adapter=True
    for batch_size in [8,16,32,64,128,256,512]:
        for learning_rate in [5e-3, 1e-4, 3e-4, 5e-4]:
            args.adapter_train_batch_size=batch_size
            args.non_adapter_learning_rate=learning_rate
            log_prefix=f"model=bert+adapter,learning_rate={learning_rate},batch_size={batch_size},"
            single_experiment(args, log_dir, log_praefix=log_prefix, test_on_valid=True)



# Single experiment method
def single_experiment(args, logs_dir, test_on_valid = False) -> None:
    experiment_start_time = time.time()

    def prediction_helper(results, mapping):
        y_pred=[p.argmax() for p in results.predictions]
        y_pred=[mapping[x] for x in y_pred]
        y_true=[mapping[x] for x in results.label_ids]

        acc=accuracy_score(y_true, y_pred)
        f1=f1_score(y_true, y_pred, average="micro")
        return y_true, y_pred, acc, f1
    data_context = load_dataset(args)
    print(data_context.df.head())


# The main method
def main() -> None:
    try:
        starttime = time.time()
        args = read_args()
        if args.experiment == "single_experiment":
            name = f"..model={args.model_name}"
            if args.subsample > 0:
                name += "..subsample"  
            log_dir = init_experiments(args, name + "..single_experiment")
            single_experiment(args, log_dir)
        elif args.experiment == "full_experiment":
            log_dir = init_experiments(args, "full_experiment")
            # full_experiment(args, log_dir)
        elif args.experiment == "search_hyperparameter":
            log_dir = init_experiments(args, "search_hyperparameter")
            # search_hyperparameter(args, log_dir)
        
        duration = (time.time() - starttime) / 60
        logging.info(f"finished in {duration:.2f} minutes")
    except Exception as e:
        logging.exception(e)


if __name__ == '__main__':
    main()
