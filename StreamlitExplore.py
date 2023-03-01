import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from Prediction import prediction
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# loading in the model to predict on the data

def showdf():
    data = load_iris()
    X=data.data
    y=data.target
    y_label=data.target_names
    df = pd.DataFrame(X, columns=data.feature_names)
    df['species'] = data.target
    df['species'] = df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica'])
    
    st.write("### Species of Iris based on Dimensions", df.head())
    return data


def main():
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
      # giving the webpage a title
    st.title("Iris Flower Prediction")
    st.header("This is Streamlit Exploration")
    st.subheader("Following is sample of data")
    data=showdf()
    st.subheader("This is testing a backend machine learning model on web app")


    
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed

      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    sepal_length = st.text_input("Sepal Length", "Type Here")
    sepal_width = st.text_input("Sepal Width", "Type Here")
    petal_length = st.text_input("Petal Length", "Type Here")
    petal_width = st.text_input("Petal Width", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
        st.success('The predicted species is {0}'.format(data.target_names[result][0]))
     
if __name__=='__main__':
    main()


