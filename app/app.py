import os
import json
import streamlit as st
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter


# @st.cache(allow_output_mutation=True)
@st.cache_resource

def load_model(cfg):
    """
    Load saved model to evalute input at app.
    :param cfg: config file
    """
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

    
    st.sidebar.subheader('About the App')
    st.sidebar.write('Text Classification App with Streamlit using a simple embeeding model')
    st.sidebar.write("This is a demo application that performs text classification. \
                     It receives a user's question about an Amazon product and predicts \
                     whether the question was asked before or after the purchase of the product")
    st.sidebar.write("Dont worry if the model predicts wrong !!!. Its solely for demo purposes.")

    st.title('Amazon Product Query Classification')
    st.write('A simple text  analysis classification app')
    st.subheader('Input the Amazon Question below')
    sentence = st.text_area('Enter your question here',height=200)
    predict_btt = st.button('predict')
    model = load_model(config)
    if predict_btt:
        raw_text = []
        raw_text.append(sentence)       
        st.info(raw_text)
        prediction = model.predict(raw_text)
        print(f"Predicted values from app: {prediction}")
        # prediction_class = prediction.argmax(axis=-1)[0]
        prediction_class = "Pre" if prediction[0] >= 0.5 else "Post"
        st.header('Prediction using Text model')
        if prediction_class == "Post":
          st.warning('Question was asked after the purchase')
        if prediction_class == "Pre":
          st.success('Question was asked before the purchase')
    
