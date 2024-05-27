import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


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
if st.button('Submit'):
    # Display the entered data
    st.write('Entered Parameters:')
    st.write('Mean Radius:', mean_radius)
    st.write('Mean Texture:', mean_texture)
    st.write('Mean Perimeter:', mean_perimeter)
    st.write('Mean Area:', mean_area)
    st.write('Mean Smoothness:', mean_smoothness)
    st.write('Mean Compactness:', mean_compactness)
    st.write('Mean Concavity:', mean_concavity)
    st.write('Mean Concave Points:', mean_concave_points)
    st.write('Mean Symmetry:', mean_symmetry)
    st.write('Mean Fractal Dimension:', mean_fractal_dimension)
    st.write('Radius Error:', radius_error)
    st.write('Texture Error:', texture_error)
    st.write('Perimeter Error:', perimeter_error)
    st.write('Area Error:', area_error)
    st.write('Smoothness Error:', smoothness_error)
    st.write('Compactness Error:', compactness_error)
    st.write('Concavity Error:', concavity_error)
    st.write('Concave Points Error:', concave_points_error)
    st.write('Symmetry Error:', symmetry_error)
    st.write('Fractal Dimension Error:', fractal_dimension_error)
    st.write('Worst Radius:', worst_radius)
    st.write('Worst Texture:', worst_texture)
    st.write('Worst Perimeter:', worst_perimeter)
    st.write('Worst Area:', worst_area)
    st.write('Worst Smoothness:', worst_smoothness)
    st.write('Worst Compactness:', worst_compactness)
    st.write('Worst Concavity:', worst_concavity)
    st.write('Worst Concave Points:', worst_concave_points)
    st.write('Worst Symmetry:', worst_symmetry)
    st.write('Worst Fractal Dimension:', worst_fractal_dimension)


