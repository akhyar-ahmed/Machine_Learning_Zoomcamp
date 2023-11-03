import re
import nltk
import torch
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader, random_split
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer


# Download NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

# Set a random seed for reproducibility
torch.manual_seed(42) 


# save all infos about training data in one place 
class DataContext():

    def __init__(self) -> None:
        self.lemmatizer = None
        self.stop_words = None
        self.tokenizer = None
        self.vectorizer = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.df=None

    # Method to convert tweets to encoded list
    def tfidf(self, texts) -> list:
        matrix = self.vectorizer.fit_transform(texts)
        return matrix.toarray()

    # preprocessing method for all tweets
    def preprocess_encode_tweets(self) -> None:
        preprocessed_tweet_ls = []
        encoded_tweet_ls = []
        for ix, row in self.df.iterrows():
            tweet = row.tweets
            # Convert to lowercase
            tweet = tweet.lower()
            
            # Remove URLs
            tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
            
            # Remove mentions and hashtags
            tweet = re.sub(r"@\w+|\#", "", tweet)
            
            # Remove special characters and punctuation
            tweet = re.sub(r"[^\w\s]", "", tweet)

            # Remove spaces
            tweet = re.sub(r'\s+', ' ', tweet)
            
            # Remove unnecessary dots
            tweet = re.sub(r'\.{2,}', '.', tweet)
            
            # Remove dots at the beginning or end of the sentence
            tweet = tweet.strip('.')
            
            # Remove spaces at the beginning or end of the sentence
            tweet = tweet.strip(' ')
            
            # Tokenization
            tokens = self.tokenizer.tokenize(tweet)
            
            # Remove stopwords
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
            
            # Lemmatization
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
            
            # Join tokens back to text
            preprocessed_tweet = " ".join(lemmatized_tokens)
            preprocessed_tweet_ls.append(preprocessed_tweet)
        
        # TFIDF enoding conversion
        encoded_tweets = self.tfidf(preprocessed_tweet_ls)
        for i in encoded_tweets:
            encoded_tweet_ls.append(i)
        
        # Create new columns in our main df
        self.df["preprocessed_tweet"] = preprocessed_tweet_ls
        self.df["encoded_tweet"] = encoded_tweet_ls  
        
        return



# Dataset can be either train, test or valid
# Use subsample to speed up processing by subsampling.
def load_dataset(args, shuffle_train = True) -> DataContext:    
    context = DataContext()
    # Tokenization
    context.tokenizer = TweetTokenizer()

    # Remove stopwords
    context.stop_words = set(stopwords.words("english"))

    # Lemmatization
    context.lemmatizer = WordNetLemmatizer()
    
    # Initialize TF-IDF Vectorizer
    context.vectorizer = TfidfVectorizer(stop_words='english', max_features=3500)

    # Read the dataset from args.dataset_path
    f = os.path.join(args.dataset_path + "twitter_sentiment_dataset.csv")
    context.df = pd.read_csv(f)

    # Preprocess and create encodings for the dataset
    context.preprocess_encode_tweets()

    # Create a custom dataset instance
    dataset = CustomDataset(context.df.encoded_tweet, context.df.sentiments)

    # Define the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for train, validation, and test sets
    context.train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = shuffle_train)
    context.valid_dataloader = DataLoader(val_dataset, batch_size = args.batch_size)
    context.test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size)

    return context



# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoded_tweet = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'text': self.encoded_tweet[idx],
            'label': self.labels[idx]
        }
        return sample



