#!/bin/bash

# The bash file runs scripts sequntially but after previous process is successfully completed.
wait
python model/train_model.py &&

python model/inference.py &&
 
streamlit run app/app.py