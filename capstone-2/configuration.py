import torch


class Config:
    def __init__(self) -> None:
        self.seed = 42
        self.exp_name = "Trans4News: Multiclass News Classifier"
        self.model_name = "bert"
        self.pre_model = "bert-base-uncased"
        self.pre_distil_model = "distilbert-base-uncased"
        self.train_PATH = "datasets/train.csv"
        self.test_PATH = "datasets/test.csv"
        self.max_length = 50
        self.num_workers = 0
        self.epochs = 5
        self.patience = 2
        self.num_classes = 4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.shuffle_train = True
        self.dropout = 0.2