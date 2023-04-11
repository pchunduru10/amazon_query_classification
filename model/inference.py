"""
Inference: The scripts loads the saved tf.keras model and executes on the test data stored in test_dir.
The final prediction (probability scores) are saved as pickle file output dir for future reference.
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
    """
    Inference script to load test data and evaluate on saved model object.
    :param cfg: config file
    :param test_dir : path to test data
    """
    
    # load saved model 
    model = tf.keras.models.load_model(os.path.join( cfg["output_dir"],"checkpoint", cfg["checkpoint_name"]))
    # vectorize_layer = model.get_layer("text_vectorization") # chck if vectorization layers is already in the model
    # Check its architecture
    # model.summary()

    # load test data
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(test_dir, 
                                                             batch_size=cfg["batch_size"])
    

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