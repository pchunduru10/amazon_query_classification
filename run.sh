#!/bin/bash

#activate virtual env
# conda init bash
# conda activate mlp
# python -m pip install --no-cache-dir -r requirements.txt

wait
python model/train_model.py
wait
python model/inference.py
wait 
streamlit run app/app.py