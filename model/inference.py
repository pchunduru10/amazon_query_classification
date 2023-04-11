"""
Inference: Add text
"""

import os 
import json
from pathlib import Path

import tensorflow as tf
from subprocess import Popen, PIPE, run

from train_model import save_output

AUTOTUNE = tf.data.AUTOTUNE


def inference(cfg: dict,
              test_dir: str):
    
    # load saved model 
    model = tf.keras.models.load_model(os.path.join( cfg["output_dir"],"checkpoint", cfg["checkpoint_name"]))
    # vectorize_layer = model.get_layer("text_vectorization")
    # Check its architecture
    # model.summary()

    # load test data
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(test_dir, 
                                                             batch_size=cfg["batch_size"])
    
    # utility for formatting the dataset from text to vectorized form
    # def vectorize_text(text, label):
    #     text = tf.expand_dims(text, -1)
    #     return vectorize_layer(text), label

    # test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)
    # print(test_ds)

    #loss, accuracy = model.evaluate(raw_test_ds)
    test_preds = model.predict(raw_test_ds) # 
    # print(f"Tests predictions from inference :{test_preds}")

    # save (test_preds)
    save_output(data =test_preds, 
            filepath = cfg["output_dir"], 
            filename = "test_predictions" )


if __name__ == '__main__':
    print("Start inference on new data")
    with open("config.json", "r") as c:
        config = json.load(c)
        print("Config read successful")

    inference(config, 
              test_dir=os.path.join(config["data_dir"],"test_data"))