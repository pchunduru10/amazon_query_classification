import os
import json
import streamlit as st
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter



# load the sentiment analysis model
# @st.cache(allow_output_mutation=True)
@st.cache_resource

def load_model(cfg):
    model = model = tf.keras.models.load_model(os.path.join( cfg["output_dir"],"checkpoint", cfg["checkpoint_name"]))
    model.summary() # included making it visible when the model is reloaded
    return model



if __name__ == '__main__':
    print(f"Build an app to demo")
    #example : Pre : How long is this cord as none of the questions or feedback with this listing is for this item?
    #example: Post : I bought 2 emotiva basx a150, planning to connect both  to an onkyo  receiver itâs this possible and how ?

    with open("config.json", "r") as c:
        config = json.load(c)
        print("Config read successful")

    st.title('XOI: Amazon Product Purchase Classification')
    st.write('A simple text  analysis classification app')
    st.subheader('Input the Amazon Question below')
    sentence = st.text_area('Enter your question here',height=200)
    predict_btt = st.button('predict')
    model = load_model(config)
    if predict_btt:
        clean_text = []
        # K.set_session(session)
        # i = text_cleaning(sentence)
        clean_text.append(sentence)
        # sequences = tokenizer.texts_to_sequences(clean_text)
        # data = pad_sequences(sequences, maxlen =  max_len)
        st.info(clean_text)
        prediction = model.predict(clean_text)
        print(f"Predicted values from app: {prediction}")
        # prediction_class = prediction.argmax(axis=-1)[0]
        prediction_class = "Pre" if prediction[0] >= 0.5 else "Post"
        st.header('Prediction using Text model')
        if prediction_class == "Post":
          st.warning('Question was asked after the purchase')
        if prediction_class == "Pre":
          st.success('Question was asked before the purchase')
    
