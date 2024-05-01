from setuptools import setup, find_packages

setup(
    name='breast-cancer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'Pillow',  # This is the Python Imaging Library used for image processing with PIL
        'tensorflow',  # Make sure to specify the version of TensorFlow you need
        'opencv-python-headless',  # OpenCV library without GUI dependencies
        'numpy',  # Required for numerical operations
    ],
)
