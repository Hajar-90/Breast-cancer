import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background
import joblib
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
st.title('Mammogram Gray Range Highlighter')

# Sidebar inputs for gray range
st.sidebar.header('Select Gray Range')
gray_lower = st.sidebar.slider('Lower Bound of Gray Range', 0, 255, 50)
gray_upper = st.sidebar.slider('Upper Bound of Gray Range', 0, 255, 150)

# File uploader for mammogram image
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Create a mask with the specified gray range
    mask = cv2.inRange(image_np, gray_lower, gray_upper)

    # Apply the mask to the image to highlight the desired areas
    highlighted_image = cv2.bitwise_and(image_np, image_np, mask=mask)

    # Convert the grayscale image and mask to 3-channel images for overlay
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Overlay the mask on the original image
    highlighted_overlay = cv2.addWeighted(image_rgb, 0.7, mask_rgb, 0.3, 0)

    # Display the original image, mask, and the highlighted image
    st.image(image_np, caption='Original Image', use_column_width=True, channels='GRAY')
    st.image(mask, caption='Mask', use_column_width=True, channels='GRAY')
    st.image(highlighted_overlay, caption='Highlighted Overlay', use_column_width=True)

# Instructions to the user
st.write("Adjust the sliders in the sidebar to change the gray range for highlighting.")

set_background('bgs/bg5.jpg')

# set title
st.title('Breast Cancer classification')


st.title('Breast Cancer Prediction Parameters Input')

# Create text inputs for each parameter
mean_radius = st.text_input('Mean Radius')
mean_texture = st.text_input('Mean Texture')
mean_perimeter = st.text_input('Mean Perimeter')
mean_area = st.text_input('Mean Area')
mean_smoothness = st.text_input('Mean Smoothness')
mean_compactness = st.text_input('Mean Compactness')
mean_concavity = st.text_input('Mean Concavity')
mean_concave_points = st.text_input('Mean Concave Points')
mean_symmetry = st.text_input('Mean Symmetry')
mean_fractal_dimension = st.text_input('Mean Fractal Dimension')
radius_error = st.text_input('Radius Error')
texture_error = st.text_input('Texture Error')
perimeter_error = st.text_input('Perimeter Error')
area_error = st.text_input('Area Error')
smoothness_error = st.text_input('Smoothness Error')
compactness_error = st.text_input('Compactness Error')
concavity_error = st.text_input('Concavity Error')
concave_points_error = st.text_input('Concave Points Error')
symmetry_error = st.text_input('Symmetry Error')
fractal_dimension_error = st.text_input('Fractal Dimension Error')
worst_radius = st.text_input('Worst Radius')
worst_texture = st.text_input('Worst Texture')
worst_perimeter = st.text_input('Worst Perimeter')
worst_area = st.text_input('Worst Area')
worst_smoothness = st.text_input('Worst Smoothness')
worst_compactness = st.text_input('Worst Compactness')
worst_concavity = st.text_input('Worst Concavity')
worst_concave_points = st.text_input('Worst Concave Points')
worst_symmetry = st.text_input('Worst Symmetry')
worst_fractal_dimension = st.text_input('Worst Fractal Dimension')

# Add a button to submit the data
if st.button('Predict'):
    # Collect the entered data
    data = np.array([
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, 
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, 
        mean_fractal_dimension, radius_error, texture_error, perimeter_error, 
        area_error, smoothness_error, compactness_error, concavity_error, 
        concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, 
        worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, 
        worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
    ], dtype=float).reshape(1, -1)
    
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make a prediction
    prediction = knn.predict(data_scaled)
    prediction_proba = knn.predict_proba(data_scaled)
    
    # Display the result
    result = 'Malignant' if prediction[0] == 1 else 'Benign'
    st.write(f'Prediction: {result}')
    st.write(f'Prediction Probability: {prediction_proba[0]}')
