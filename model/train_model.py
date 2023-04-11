"""
This code is meant to be useful to save time getting a simple classifier built
after the data have been saved in the format expected by the data loader function:
    tf.keras.utils.text_dataset_from_directory

This code is mostly a copy-paste plus some rearranging from the TensorFlow docs
tutorial:
    https://www.tensorflow.org/tutorials/keras/text_classification

Feel free to look over that tutorial and use the code there to help with this work,
if needed. If you want to change anything here, feel free. Import whatever packages
you decide you want to use.

There are a few places with TODO in the comments for you to decide how / where to
save some of the output, log some info to the console, or run an evaluation. This
part of the pipeline is mostly done for you, so shouldn't take a lot of time.


Notes: 
  - This code was tested in TF 2.11.0
  - Remember to use a small subset of the full dataset so this code runs quickly.
    20K training, 5k test should be good for our purposes here. If it's slow, you
    can further downsample or reduce the epochs. The point is to get something that
    trains a model and saves it, not to get an optimized model.
"""
import os 
import json
import psutil
from pathlib import Path

import pandas as pd
import pickle
import datetime
from typing import Callable, Any, Iterable

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from subprocess import Popen, PIPE, run

AUTOTUNE = tf.data.AUTOTUNE


def load_data(
    train_directory: str,
    val_frac: float = 0.2,
    batch_size: int = 64,
    seed: int = 41924,
):
    """Loads data using keras utility from the provided training and test set
    directories. Assumes file format described in the documentation.
    :param train_directory: path to train data
    :param val_frac: split for validation from train data
    :batch_size: batch size for training
    :seed : seed
    """
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        train_directory,
        batch_size=batch_size,
        validation_split=val_frac,
        subset="training",
        seed=seed,
    )

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        train_directory,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=seed,
    )
    return raw_train_ds, raw_val_ds 


def build_model(max_features: int, embedding_dim: int):
    """Creates a simple Model object to be trained on the text dataset.
    :param max_features: maximum features to train on.
    :param embedding_dim : cfg["embedding_dim"]

    """
    model = tf.keras.Sequential(
        [
            layers.Embedding(max_features + 1, embedding_dim),
            layers.Dropout(0.5),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1),
        ]
    )
    #print model summary in the  console (std out)
    model.summary()
    return model


def train_model(
    cfg: dict,
    train_dir: str
    ):
    """Loads data and trains the simple classifier model.
    Returns the text vectorizer, model object, and label-name mapping
    
    :param cfg: config file.
    :param train_dir: path to train data.    

    """

    # load data
    raw_train_ds, raw_val_ds = load_data(train_dir, 
                                        batch_size=cfg["batch_size"])

    # figure out what 0 and 1 correspond to in the text labels
    class_name_map = dict(zip(range(2), raw_train_ds.class_names))

    # create text vectorization layer
    vectorize_layer = layers.TextVectorization(
        standardize="lower",
        max_tokens=cfg["max_features"],
        output_mode="int",
        output_sequence_length= cfg["sequence_length"],
    )

    # adapt the layer to the input text
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    # get the model (except the vectorizer and final sigmoid activation)
    model = build_model(cfg["max_features"], cfg["embedding_dim"])
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer="adam",
        metrics="accuracy",
    )

    # utility for formatting the dataset from text to vectorized form
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # prep datasets
    train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)

    # train model
    history = model.fit(train_ds, validation_data=val_ds, epochs= cfg["max_epochs"])
    save_output(data = history.history, 
                filepath = cfg["output_dir"], 
                filename = "train_history_dict" )

    return vectorize_layer, model, class_name_map


def save_model(model_object: tf.keras.Model, output_dir: str):
    """Save the final model in the output path.

    :param model_object: final model
    :type model_object: tf.keras.Model
    """

    # creat directory is not already exists
    Path(os.path.join(output_dir,"checkpoint")).mkdir(parents=True, exist_ok=True)

    # now = datetime.datetime.now()
    filename = "text_model" #now.strftime('model_%Y%m%d_%H%M%S')
    print(f"Saving the final model")
    model_object.save(os.path.join(output_dir,"checkpoint",filename))



def save_output(data: Any, filepath: str, filename: str = "train_history_dict"):
    """Save output as pickle file in given location.

    :param data: data to save (predictions or history)
    :type data: Any
    """
    
    with open(os.path.join(filepath, f"{filename}.p"), 'wb') as file_pi:
        pickle.dump(data, file_pi)


def main(cfg: dict, train_dir: str):
    """Main file to train the text classfication model.

    :param train_dir: path to train data
    :type train_dir: str
    :param cfg: config file loaded
    :type output_dir: python dictionary
    """
    # check for dir path and if not exists create
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # train the model on the data
    vectorize_layer, model, class_name_map = train_model(cfg=cfg,
                                                        train_dir = train_dir)

    # TODO: save class name map as json file in the output_dir
    # class_name_map are dictionaries
    with open(os.path.join(cfg["output_dir"],'class_name_map.json'), 'w') as fp: 
        json.dump(class_name_map, fp)

    # put model into exportable form and save
    export_model = tf.keras.Sequential([vectorize_layer, model, layers.Activation("sigmoid")])
    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )

    # save the model which can be loaded in inference.py to make predicitons
    save_model(export_model, cfg["output_dir"])


def on_terminate(proc):
    print(f"process {proc} terminated")


if __name__ == "__main__":
    """
    Main script to execute.
    """
    with open("config.json", "r") as c:
        config = json.load(c)
        print("Config read successful")

    print(f"The current working dir is {os.getcwd()}")
    
    p = Popen(["python", "model/load_data.py"]) 
    while True: 
        gone, alive = psutil.wait_procs([p], timeout=3, callback=on_terminate) 
        if len(gone)>0:
            break
    
    main(config,
        train_dir = os.path.join(config["data_dir"],"training_data"))
    
