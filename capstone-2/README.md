# Trans4News: Multiclass News Classifier
---------
## Table of Contents
- [Project Overview & Dataset](#project-overview-dataset)
- [Model Selection Process](#model-selection-process)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
- [Directory description](#directory-description)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## [Project Overview & Dataset](#project-overview-dataset)

The Trans4News project aims to develop a multiclass news classifier utilizing the AG News Topic Classification dataset. This dataset, comprising over 1 million news articles, has been meticulously curated from 2000 diverse news sources by the academic news search engine, ComeToMyHead. The dataset spans more than a year of activity, starting from July 2004.

### Dataset Details:

**Source:** AG Corpus of News Articles

**Provider:** ComeToMyHead Academic News Search Engine

**Construction:** Xiang Zhang (xiang.zhang@nyu.edu) curated the AG's news topic classification dataset from the original corpus.

**Benchmark Usage:** The dataset is employed as a text classification benchmark, specifically in the paper titled "Character-level Convolutional Networks for Text Classification" (Xiang Zhang, Junbo Zhao, Yann LeCun, Advances in Neural Information Processing Systems 28 - NIPS 2015).

**Dataset Composition:** The AG's news topic classification dataset consists of the four largest classes chosen from the original corpus. Each class comprises 30,000 training samples and 1,900 testing samples, resulting in a total of 120,000 training samples and 7,600 testing samples.

**Label Information:** The file classes.txt contains a list of classes corresponding to each label.

**Data Format:** The dataset is provided in two comma-separated values (CSV) files - train.csv and test.csv. Each file contains three columns: class index (1 to 4), title, and description. Titles and descriptions are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed by an "n" character (i.e., "\n").

### Purpose of the Project:

The Trans4News project aims to implement a multiclass news classifier using advanced natural language processing (NLP) techniques `transfomer models` (`BertForSequenceClassifier`, `DistilBertForSequenceClassifier`). The classifier will be trained on the AG's news topic classification dataset to effectively categorize news articles into predefined classes. The primary purposes of the project include:

**Text Classification Benchmarking:** Evaluate the performance of the classifier against the AG's news topic classification dataset, serving as a benchmark for multiclass text classification tasks.

**Research in NLP:** Contribute to the field of natural language processing by comparing and implementing advanced models for news article classification.

**Open-Source Contribution:** Provide an open-source solution for multiclass news classification, encouraging collaboration and further advancements in the field.

**Education and Learning:** Create a learning resource for developers, researchers, and enthusiasts interested in NLP, text classification, and machine learning.

![Title_Image](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/50096da1-6e05-4331-928e-407186538533)

Credit: [Bing Chat](https://www.bing.com/?FORM=Z9FD1)

There's nothing much to tune for such architecture model. I first tried scheduling to adjust learning-rate dynamically but I found it time consuming so I only stick with some static learning rates to tune. Another thing is, I also tried by adding a dropout layer before calculating the final outputs. But found that the result don't change much. So I also remove that part from final training.`trans4news-multiclass-news-classifier.ipynb` file has everything written step by step. I implemented early-stopping technique on validation-loss, summarywriting from tensorboard to minitor the model results, and also model checkpointing to get the best model on the base of validation loss. Which means Validation-loss defines the perforfence of the best model for each architecture. After the training DistilBertForSequenceClassifier show better result. So I stick with that architecture for furthure proceeding. Later I convert it an anssemble everything in **train.py**. So you can follow that notebook and that script to evaluate my project. Using the best trained model I make predictions on whether it can detect the classes(`class_labels.json`) of a news article. Finally, I containerise this application, publish it to docker-hub, and deployed it to cloud using AWS Fargate, ECS, ECR.


## [Model Selection Process](#model-selection-process)

As mentioned above the model selection process involves choosing a best hyper-parameter combination for my dataset that best fits the multi-class classification task. Potential hyperparameter includes:

- compare results between `BertForSequenceClassifier` and `DistilBertForSequenceClassifier`
- adjusting learning rate (see `trans4news-multiclass-news-classifier.ipynb`'s model)
- adjusting dropout rate (I tried with several dopout values [0.1, 0.2, 0.3]. But results don't change much and it's also time consuming so I later drop this.)

I use Adam optimizer to calculate the loss and CrossEntropy for criterion. So whichever configuration give best validation loss, I choose that model for final training. 
- See `configuration.py`, which shows you all the configuration details of the best model (which performed well).




## [Dependencies](#dependencies)

The project requires the following dependencies to be installed:

```
Conda
Docker
```


## [Workflow](#workflow)

To replicate the project environment, follow these steps:
### 1. Clone the repository:

```bash
   - git clone -b v1.1 https://github.com/akhyar016/mlzoomcamp_homeworks.git
   - cd capstone-2
```

### 2. Setting up the environment:

```bash
    pip install pipenv
    pipenv install
    pipenv shell
```

Now, you have a virtual environment with all the necessary dependencies installed, allowing you to run the project code seamlessly.

### 3. Running `notebooks/trans4news-multiclass-news-classifier.ipynb` 

This notebook outlines the entire investigation and consists of the following steps [🚨 Skip this step, if you want to directly want to use the final configuration for training and/or final model for predictions]:

- Package installation
- Configuration class
- Set random seed for reproducibility
- Data loading and preprocessing (Which includes, EDA, Tokenization, Dataset creation, and Dataloader creation for all train, valid, test set)
- Model training [and hyper-parameter tuning]
- Saving the best model(`best_model_DistilBertForSequenceClassification_1e-05.bin`) in the [models](./models) directory
- Evaluate the best model on test set.
- Making predictions using the saved model (`local_test.py`).

### 4. Training model

We encode our best model (`best_model_DistilBertForSequenceClassification_1e-05.bin`) inside the `train.py` file which can be run using:
```
python -m train
```
You can see all kind of outputs including checkpoints in the `output/` folder.


### 5. Making predictions

I have written a REST API code (`app.py`) for serving the model by exposing the port:8080, which can be run using:

```
uvicorn app:app --host=0.0.0.0 --port=8080
```

After running it you can go to `http://localhost:8080/predict` and can use this to make an example prediction:

![Trans4News_predictions_1](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/c34fb51e-cedc-4764-a7a2-8dd531627427)


Credit: [Akhyar Ahmed](https://akhyar-ahmed.github.io/portfolio/)

### 6. Containerizing the model

Run the `Dockerfile` using [make sure that the docker daemon is running?] to build the image `projects-capstone-2.0`:

```bash
docker build --pull --rm -f "capstone-2\DockerFile" -t mlzoomcamphomeworks:projects-capstone-2.0 "capstone-2"
```

Once the image is built, we need to expose the container port (8080) to the localhost port (8080) using:

```bash
docker run --rm -it -p 8080:8080/tcp mlzoomcamphomeworks:projects-capstone-2.0
```

Alternatively,
Pull the Docker Image from Docker Hub: Run the following command in your terminal or command prompt to pull the Docker image from Docker Hub:
```bash
docker pull akhyarahmed/mlzoomcamp:projects-capstone-2.0
docker run --rm -it -p 8080:8080/tcp akhyarahmed/mlzoomcamp:projects-capstone-2.0
```



We can now make a request in exactly the same way as Step 5:

```bash
python local_test.py
```
![local_test](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/dd832eb2-e9c3-4aa9-a8f2-3f11a4fffeff)



🚨 You can use some more test data from `datasets/test.csv` file. These are unseen news. Class Index is the class number of each news. So you can test with these news and match the results with the predicted result with the help of `class_label.json`.


### 7. Cloud Deployment

After quantization the best model I deployed it into AWS. I used AWS ECR to upload the docker image, fargate to create the task, and ECS to run the task. To do all I need AWS CLI first. So first install AWS CLI from their official AWS website. Than follow the below steps:

#### 7.1 AWS Configuration:
Configure with your own aws credintials. You can find all your credintials into AWS IAM service. 

```bash
aws configure
```

![aws_config](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/ac9b45a7-1be0-4b3e-a617-38d6aff4b5f6)

### 7.2 Upload your docker image to AWS ECR:
* First enable ECR and then login into ECR from your **user_id**
```bash
aws ecr get-login-password --region your-time-zone-region | docker login --username AWS \
--password-stdin your-aws-user-id.dkr.ecr.your-time-zone-region.amazonaws.com
```
* Secondly build your docker image and upload it to ECR
```bash
docker tag your-repo-name:tags your-aws-user-id.dkr.ecr.your-time-zone-region.amazonaws.com/your-aws-registry-name:tags

docker push your-aws-user-id.dkr.ecr.your-time-zone-region.amazonaws.com/your-aws-registry-name:tags
```

![aws_login_docker_tag_push](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/b0143af2-2957-4931-ad54-06ef6df7af65)



### 7.3 Create AWS ECS cluster:
Create a ECR cluster with a proper name.
```bash
aws ecs create-cluster --cluster-name your-cluster-name
```

![aws_create_cluster](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/4c05b35f-4c63-4101-b3f3-48e63a0f227a)

### 7.4 Write, register, run, and expose a task definition:
* We need a task definition to deplpoy and expose our docker image. In that file we also write about the port we want to use, how much memory we need, etc. I wrote for my windows system and my task is compatible for Fargate and EC2. Here is my task definition (lets name it `task-definition.json`):
```bash
{
  "containerDefinitions": [
      {
          "command": ["New-Item -Path C:\\inetpub\\wwwroot\\index.html -Type file -Value '<html> <head> <title>Amazon ECS Sample App</title> <style>body {margin-top: 40px; background-color: #333;} </style> </head><body> <div style=color:white;text-align:center> <h1>Amazon ECS Sample App</h1> <h2>Congratulations!</h2> <p>Your application is now running on a container in Amazon ECS.</p>'; C:\\ServiceMonitor.exe w3svc"],
          "entryPoint": [
              "powershell",
              "-Command"
          ],
          "essential": true,
          "cpu": 2048,
          "memory": 4096,
          "image": "your-aws-user-id.dkr.ecr.your-time-zone-region.amazonaws.com/mlzoomcamp:capstone-2.0",
          "name": "sample_windows_app",
          "portMappings": [
              {
                  "hostPort": 8080,
                  "containerPort": 8080,
                  "protocol": "tcp"
              }
          ]
      }
  ],
  "memory": "4096",
  "cpu": "2048",
  "networkMode": "awsvpc",
  "family": "your-task-family-name",
  "executionRoleArn": "arn:aws:iam::your-aws-user-id:role/ecs-task",
  "runtimePlatform": {"operatingSystemFamily": "WINDOWS_SERVER_2019_CORE"},
  "requiresCompatibilities": ["FARGATE"]
}

```
🚨Also re confirm that your `executionRoleArn` has permission to receive requests from AWS services. You can handle it in AWS ECS dashboard.

* Second thing is we need to register this task definition to ECS. 🚨 Don't forget to add `file://` before your file path, cause this is a syntax for AWS CLI.
```bash
aws ecs register-task-definition --cli-input-json file://your-task-definition-file-path/your-task-definition-file-name.json
```

![aws_register_task_definition](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/30471a1c-9fcc-475a-90d1-a687b433d5f3)

* Run your registered task definition. To do take we need task-family-name from `task-definition.json`'s `family` attribute, launch-type should be fargate, netwrok configuration as I used awsvpc netwrokmode. For network configuration we need public subnet ip and security group. Both can be found from your AWS IAM console. Just remember to take available public ip's from `your-time-zone-region`.
```bash
aws ecs run-task --cluster your-cluster-name --task-definition your-task-family-name --count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-1234id],securityGroups=[sg-1234group],assignPublic=ENABLED}"
```


* Expose that running task.

```bash
aws ecs create-service --cluster your-cluster-name --service-name any-name-for-this-service --task-definition take-it-from-above --desired-count 1 --launch-tyoe FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-1234id],securityGroups=[sg-1234group],assignPublic=ENABLED}"
```


### 7.5 Check your service and clean up everything
Once your service is up and running (you can check it in your AWS ECS console) you will see a public ip or DNS under your ECS service with that you can access your application.
```bash
http://<public-ip-or-dns>:<your-app-port>
```
🚨 Remember to remove all the created services and clusters to avoid getting charges. To do that just run below commands in AWS CLI

* Delete the service we created for our task.
```bash
aws ecs delete-service --cluster your-cluster-name --service your-given-serive-name-from-above --force
```

* Delete the FARGATE cluster
```bash
aws ecs delete-cluster --cluster your-cluster-name
```


## [Directory description](#directory-description)

🚨 `class_labels.json` has the name of all clasees.

🚨 `Pipfile` and `Pipfile.lock` are the used environments you can also use them.

🚨 `preprocessomg.py` has all the EDA and preprocessing codes.

🚨 `create_light_model.py` script to convert our the best_model.h5 to the best_model.tflite. So that I can use it for cloud deployment `quantized_best_model.pt`.

🚨 `models/` you can find the tflite model their.

## [Contributors](#contributors)
[Akhyar Ahmed](https://akhyar-ahmed.github.io/portfolio/)

## [Acknowledgments](#acknowledgments)
* [Alexey Grigorev](https://github.com/alexeygrigorev)
* [DataTalks.Club](https://datatalks.club/)
