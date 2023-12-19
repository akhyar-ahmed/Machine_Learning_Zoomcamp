# PhytoPhinder: The Leaf Disease Detective
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

PhytoPhinder is an innovative plant disease detection system that harnesses the power of Convolutional Neural Networks (CNNs). It diagnoses diseases in plants based on leaf images with remarkable accuracy. The system is built using PyTorch, a leading deep learning framework, and is deployed as a FastAPI application in a Docker container, ensuring scalability and ease of use. The project utilizes a comprehensive dataset of leaf images from Kaggle, known as the Leaf Disease Detection [Dataset](https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset/data). 

This dataset provides a diverse range of leaf images, covering various species and disease states, which aids in training a robust and versatile model. The dataset comprise with approximate ~90k of images (train ~70k, test ~20k). The dataset only provide 38 types of disease and healthy images of several plants. EDA has been done in `notebooks/phytophinder-the-leaf-detective.ipynb` notebook. You can see all the types of plant sample (healty and non-healthy) images. Which means this problem is a multi-class classification problem and we have 38 classes to classify. I also created `small_dataset` (you can download it from [here](https://drive.google.com/file/d/1noWajl7rgjk_84uHx2OkdcQ-cAth8nuZ/view?usp=sharing)). I used this dataset in `notebooks/notebook_1.ipynb` to run the project locally. You can use this dataset to run the project locally. Don't worry it has all equally distributed images from all classes. 

![Title_Image](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/647f40f9-819b-4c8e-8653-f7af0c02a945)

Credit: [Bing Chat](https://www.bing.com/?FORM=Z9FD1)

I hyper-parameter tune it with differnt CNN models (`notebook_1.ipynb`, and `phytophinder-the-leaf-detective.ipynb`) with different inner-layers. Validation-loss defines the perforfence of the best model. Where `phytophinder-the-leaf-detective.ipynb` is the final version of my code. Later I convert it an anssemble everything in **train.py**. So you can follow that notebook and that script to evaluate my project. Using the best trained model I make predictions on whether it can detect the classes(disease or healthy) of a plant leaf. Finally, I containerise this application and publish it to docker-hub.


## [Model Selection Process](#model-selection-process)

As mentioned above the model selection process involves choosing a best hyper-parameter combination for my dataset that best fits the multi-class classification task. Potential hyperparameter includes:

- with/without an extra inner-layer (see `notebook_1.ipynb`'s model and `phytophinder-the-leaf-detective.ipynb`'s model for better comparison)
- adjusting learning rate (see `phytophinder-the-leaf-detective.ipynb`'s model)
- adjusting dropout rate (see `phytophinder-the-leaf-detective.ipynb`'s model)
- adjusting the size of inner layer (see `phytophinder-the-leaf-detective.ipynb`'s model)

I use Adam optimizer to calculate the loss. So whichever configuration give best validation loss, I choose that model for final training. 
- See `notebooks/best_model_config.json`, which shows you all the configuration details of the best model (which performed well).




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
   - cd capstone-1
```

### 2. Setting up the environment:

```bash
    pip install pipenv
    pipenv install
    pipenv shell
```

Now, you have a virtual environment with all the necessary dependencies installed, allowing you to run the project code seamlessly.

### 3. Running `notebooks/phytophinder-the-leaf-detective.ipynb` 

This notebook outlines the entire investigation and consists of the following steps [🚨 Skip this step, if you want to directly want to use the final configuration for training and/or final model for predictions]:

- Data loading
- Data cleaning and preparation
- Exploratory data analysis
- Setting up a ImageDataGenerator
- Model evaluation [and hyper-parameter tuning]
- Saving the best model and encoders [in the [models](./capstone-1/models) directory]
- Making predictions using the saved model

### 4. Training model

We encode our best model (`notebooks/best_model_config.json`) inside the `train.py` file which can be run using:
```
python -m train
```
You can see all kind of outputs including checkpoints in the `output/` folder.


### 5. Making predictions

I have written a REST API code (`app.py`) for serving the model by exposing the port:8001, which can be run using:

```
uvicorn app:app --host=0.0.0.0 --port=8001
```

After running it you can go to `localhost:8001/predict` and can use this to make an example prediction:

![Screenshot 2023-12-18 215159](https://github.com/akhyar-ahmed/Machine_Learning_Zoomcamp/assets/26096858/8d27cd43-94e4-4f76-a7fa-7b1c27eb768d)

Credit: [Akhyar Ahmed](https://akhyar-ahmed.github.io/portfolio/)

### 6. Containerizing the model

Run the `Dockerfile` using [make sure that the docker daemon is running?] to build the image `projects-capstone-1.0`:

```bash
docker build --pull --rm -f "capstone-1\DockerFile" -t mlzoomcamphomeworks:projects-capstone-1.0 "capstone-1"
```

Once the image is built, we need to expose the container port (8001) to the localhost port (8001) using:

```bash
docker run --rm -it -p 8001:8001/tcp mlzoomcamphomeworks:projects-capstone-1.0
```

Alternatively,
Pull the Docker Image from Docker Hub: Run the following command in your terminal or command prompt to pull the Docker image from Docker Hub:
```bash
docker pull akhyarahmed/mlzoomcamp:projects-capstone-1.0
docker run --rm -it -p 8001:8001/tcp akhyarahmed/mlzoomcamp:projects-capstone-1.0
```



We can now make a request in exactly the same way as Step 5:

```
python local_test.py
# ["Predicted class: Tomato___Tomato_Yellow_Leaf_Curl_Virus"] 
```
🚨 You can use some more test data from `images_for_test/` folder. These are unseen images. Image name is the class name of each image. So you can test with these pictures and match the results with the predicted result.

## [Directory description](#directory-description)

🚨 `class_labels.json` has the name of all clasees.

🚨 `Pipfile` and `Pipfile.lock` are the used environments you can also use them.

🚨 `eda_and_preprocessomg.py` has all the EDA and preprocessing codes.

🚨 `plots/` folder has all the figures, which were created during EDA or image content analysis.

🚨 `create_light_model.py` script to convert our the best_model.h5 to the best_model.tflite. So that I can upload it to github.

🚨 `models/` you can find the tflite model their.

## [Contributors](#contributors)
[Akhyar Ahmed](https://akhyar-ahmed.github.io/portfolio/)

## [Acknowledgments](#acknowledgments)
* [Alexey Grigorev](https://github.com/alexeygrigorev)
* [DataTalks.Club](https://datatalks.club/)
