from setuptools import setup, find_packages

setup(
    name='app.py',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'numpy',
        'tensorflow-cpu', 
    ],
)

