# Twitter Sentiment Classification (ML Zoomcamp Mid-term Project)
---------

## Problem Description

The Twitter Sentiment Classification project aims to build a robust model for determining the sentiment tone (positive or negative) of text data from Twitter. The goal is to create an accurate classifier that can understand the sentiment expressed in tweets, allowing for applications in sentiment analysis, customer feedback processing, and social media monitoring.

## Context

In the age of social media, understanding public sentiment is crucial for businesses, researchers, and policymakers. This project addresses the need for an effective sentiment analysis tool specific to Twitter, one of the most influential social media platforms. By classifying tweets as positive or negative, we can gain insights into public opinion, brand perception, and societal trends.

## Dataset and Exploratory Data Analysis (EDA)

The project utilizes a labeled dataset **([train.csv](https://www.kaggle.com/competitions/twitter-sentiment-analysis2/data?select=train.csv))** containing Twitter text and corresponding sentiment labels (positive or negative). Before building the model, an Exploratory Data Analysis (EDA) was conducted. EDA involved:

- Data visualization to understand the distribution of positive and negative tweets.
- Text preprocessing to clean and prepare the text data for model training.

## Model Selection Process

The model selection process involves choosing a machine learning algorithm that best fits the sentiment classification task. Potential algorithms include:

- Logistic Regression
- XGBoost Classifier
- Random Forest Classifier
- Multi-layer Perceptron Classifier

The model selection process also includes:

- Hyperparameter tuning: Adjusting parameters for optimal model performance.
- Cross-validation: Assessing model generalizability using k-fold cross-validation.
- Performance evaluation: Measuring model performance using Accuracy and F1-Score.

## Instructions for Dependencies and Virtual Environment Setup

To replicate the project environment, follow these steps:

1. Clone the repository:
    ```bash
   - git clone -b v1.0 https://github.com/akhyar016/mlzoomcamp_homeworks.git
   - cd src
2. Install Pipenv:
    ```bash
    pip install pipenv
3. Install Dependencies:
    ```bash
    pipenv install
4. Activate Virtual Environment:
    ```bash
    pipenv shell
Now, you have a virtual environment with all the necessary dependencies installed, allowing you to run the project code seamlessly.

## Project Folder Structure

This document provides an overview of the folder structure of the project.
```
src/
│
├── asset/
│
├── ipynb_files/
│
├── logs/
│
├── models/
│
├── dataloader.py
|
├── main.py
|
├── model.py
|
├── train.py
|
├── utils.py
|
├── DockerFile
|
├── Pipfile
|
├── Pipfile.lock
|
README.md
```

### Descriptions

* **src/**: The source code for the project is organized here. You will find utility scripts, model code, and other project-related code files.
* **src/asset/**: This directory contains subdirectories for storing processed datasets used in the project.
* **src/ipynb_files/**: Jupyter notebooks used for exploratory data analysis (EDA) and experimentation are stored here.
* **src/models/**: Stored a best estimated model.
* **src/logs/**: Contains logger file for each experiment.
* **src/DockerFile**: Docker file provide a tool that creates a docker image. 
* **src/Pipfile** and **src/Pipfile.lock**: Pipenv environment file.
* **src/train.py**: Train file which trains multiple classifiers and provide results under that logger **logs/** folder.
* **src/model.py**: Create instance of every models and also have a method for hyperparameter tuning.
* **src/utils.py**: Create **logs/** folder, argument parser and sub folder for each run. It also has method to store results of each experiment.
* **src/dataloader.py**: Which splits the dataset and creates torch dataloader. 
* **src/main.py**: The main file for our RESTful app. Which initiate a RESTFUL API.
* **src/TRAINME.md**: 
* **README.md**: This file you're currently reading provides an overview of the folder structure, how to clone this project,  and install the environment.

## Containerization
1. Pull the Docker Image from Docker Hub: Run the following command in your terminal or command prompt to pull the Docker image from Docker Hub:
```bash
docker pull <your-docker-image-name>:<tag>
```



## Conclusion

The Twitter Sentiment Classification project provides a valuable tool for analyzing sentiments in Twitter data. By following the described process, understanding the problem context, and replicating the virtual environment, users can gain insights into Twitter sentiments, contributing to various applications across industries.