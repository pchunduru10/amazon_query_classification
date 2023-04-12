# XOI Technical Challenge
![alt text](https://xoi.io/wp-content/uploads/2023/01/XOi_Logo.svg)

## Overview
The aim of this project is to create a basic model pipeline which encompasses the key components of a machine learning framework. The process involves importing and transforming the raw data to make it compatible with a machine learning model, training a text classification model using the processed data, and saving the final model. To enhance user experience and facilitate the deployment of the ML model, a user-friendly application will also be developed.

1. Data Processing using raw csv.
2. Train the ML-model - Binary classification model to predict whether the product based question was asked before or after the purchase.
3. Batch inference on test data by loading the model.
4. Create a streamlit app for user testing.
5. Build a custom docker image for the entire ML-model and App.

A config.json is included that takes in all the parameters necessary to execute the model.

## 1. Data Processing
The script for loading data involves reading a large CSV file from a URL and converting it into a Pandas dataframe. To simplify the demonstration, a technique called stratified downsampling was utilized to decrease the dataset to 25,000 samples for both the training and testing sets.
A data dir is created when executing the docker container.

## 2. Train the ML-Model
In order to prioritize the creation of a pipeline and the deployment of the app, the main focus is on containerizing the pipeline instead of improving the accuracy of the model. Therefore, a pre-existing model was used for training, and the script `train_model.py` was developed. This script begins by loading `load_data.py` and waits for it to finish before building and training the model. The resulting output from the model, including logs, the class map, and the final checkpoint, are all saved in a directory named "output," which is created during the execution of the script. It's worth noting that neither feature engineering nor model optimization are covered in this demo.

## 3. Model inference on test data
`inference.py` loads the saved tensorflow model (along with weights and trained params) and generates preediction on test set (saved as pickle object in output directory)

## 4. Create a streamlit app for user testing.

The script `app/app.py` constructs a basic user interface utilizing the Streamlit library. It loads the saved model and prompts the user for input. The input can be in the form of a raw string, which is then processed during the prediction phase to determine the category of the question asked.

## 5. Containerization of the ML-model
The entire machine learning model is executed through a bash file called `run.sh`. This file is responsible for training the model, making inferences on the test set, and building the app. The `Dockerfile` contains all of the necessary components to build the image and execute the bash file.

A requirements file is created to install the necessary dependencies

## Instructions to build image and execute the model.

**Prerequisite** - Make sure you local system has Docker already installed.

**Clone the Repo** - Since this is a private repo, Github may ask you to create Personal Access Token to clone the repo. Make sure you are familiar with and already have setup shh keys and personal access token before starting.

Clone the repo (using HTTPS)/
`git clone  https://github.com/pchunduru10/xoi_technical_challenge.git`

Make the `run.sh` executable by running the coomand `chmod +x run.sh`  [ TODO : CAN BE ADDED IN DOCKERFILE]

Build the docker image using the following command

`sudo docker build -t <image_name> -f Dockerfile .` 

To create and run the docker container.

`sudo docker run -ti <image_name>`

*sudo* might be optional if you are not running as root.

Once the container runs it will generate a Network-URL to the app. Follow the link to give user input for the model. [You can also see the prediction value in the stdout]



