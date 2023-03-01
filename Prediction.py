import pickle
def prediction(sepal_length, sepal_width, petal_length, petal_width):  
    pickle_in = open('C:/Users/v-sarshah/Downloads/Streamlit Explore/classifier.pkl', 'rb')
    classifier = pickle.load(pickle_in)
   
    prediction = classifier.predict(
        [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
    print(prediction)
    return prediction