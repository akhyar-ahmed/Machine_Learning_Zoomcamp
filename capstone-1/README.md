# PhytoPhinder: The Leaf Disease Detective
---------
## Table of Contents
- [Project Overview](#project-overview)
- [Getting started immediately](#getting-started)
- [Datasets](#datasets)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
- [Directory structure](#dirctory-structure)
- [Models](#models)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contributions and Feedback](#contributions-and-feedback)

## [Project Overview](#project-overview)
### Description
PhytoPhinder is an innovative plant disease detection system that harnesses the power of Convolutional Neural Networks (CNNs). It diagnoses diseases in plants based on leaf images with remarkable accuracy. The system is built using PyTorch, a leading deep learning framework, and is deployed as a FastAPI application in a Docker container, ensuring scalability and ease of use. The project utilizes a comprehensive dataset of leaf images from Kaggle, known as the Leaf Disease Detection [Dataset](https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset/data). This dataset provides a diverse range of leaf images, covering various species and disease states, which aids in training a robust and versatile model.

![Title_Image](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/e41a757d-eacc-4966-9bf1-7b2fb91148c3)


### Importance
Plant diseases can lead to significant reductions in crop yield and quality, resulting in economic losses and food security issues. Early and accurate detection of plant diseases can enable timely intervention, preventing widespread damage and ensuring the health of our ecosystems.

### Purpose
The purpose of PhytoPhinder is to provide an accessible, accurate, and rapid tool for plant disease diagnosis. By automating the process of disease detection, it can assist farmers, gardeners, and researchers in monitoring plant health and implementing effective treatment strategies. This project aims to contribute to sustainable agriculture practices and enhance our ability to safeguard plant biodiversity.



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
   - git clone -b v1.1 https://github.com/akhyar016/mlzoomcamp_homeworks.git
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
docker pull akhyarahmed/mlzoomcamp:sentiment-analysis-v1.0
```

2. Run the Docker Container on Port 8001:
Run the following command to start a Docker container based on the pulled image and map port 8001 on your host system to the container's port:
```bash
docker run --rm -it -p 8001:8001/tcp akhyarahmed/mlzoomcamp:sentiment-analysis-v1.0
```


## REST API
To interact with the FAST API,
1. Using Docker, just simply pull and run docker image using above commands then go to: 
```bash
www.localhost:8001/predict
```
2. Go to **src** directory (you can find directory tree in above) and run following command:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```
then go to this link:
```bash
www.localhost:8001/predict
```

## Conclusion

The Twitter Sentiment Classification project provides a valuable tool for analyzing sentiments in Twitter data. By following the described process, understanding the problem context, and replicating the virtual environment, users can gain insights into Twitter sentiments, contributing to various applications across industries.
