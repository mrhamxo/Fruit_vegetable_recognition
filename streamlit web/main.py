import streamlit as st
import numpy as np 
import tensorflow as tf

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('../model/trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size = (64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # converting single into batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) # return index of maximum element

# sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox('Select Page', ["Home", "About Project", "Prediction"])

# Home Page
if(app_mode == "Home"):
    st.header('FRUITS & VEGETABLES RECOGNITION SYSTEM')
    image_path = "home_image.jpg"   
    st.image(image_path)

# About Project Page
elif(app_mode == "About Project"):
    st.header('ABOUT PROJECT')
    st.subheader('About Dataset')
    st.text('This dataset contains images of the following food items:')
    st.code('fruits:- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.')
    st.code('vegetables:- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.')
    st.subheader('Content')
    st.text('This dataset contains three folders:')
    st.text('1. Train (100 images each)')
    st.text('2. Test (10 images each)')
    st.text('3. Validation (10 images each)')
    
# Prediction Page
elif(app_mode == "Prediction"):
    st.header('MODEL PREDICTION')
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
       st.image(test_image, width = 4, use_column_width = True)

       # create Prediction button
    if(st.button('Predict')):
        st.balloons()
        st.write('our prediction')
        result_index = model_prediction(test_image)
        # Reading Labels
        with open('labels.txt') as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[ :-1])
        #st.write(label)
        st.success("Model is Predicting  it's a {}" . format(label[result_index]))  