import streamlit as st
import pandas as pd
import numpy as np

from utils import *
from predict import *
from retrain_model import *
import os
def app():

    @st.cache
    def get_pdf(filepath):
        pages_pdf = extract_clauses_from_pdf(filepath)
        clauses_pdf = preprocess_clauses_pdf(pages_pdf)
        return clauses_pdf
    @st.cache
    def get_docx(filepath):
        return get_text_from_docx(filepath)

    st.title('Retrain EULA Classification model')
    
    st.write("Welcome to the EULA Retraining App. This app allows you upload training data to retrain the Machine Learning Model based on user input.")
    st.write("The way this is accomplish is by retraining the model using Transfer Learning. Transfer Learning is when you utilize the model's pre-extisting knowledge, and improve it based on new input.")
    st.write(" ")
    st.subheader("To get started, enter the file path to directory of dataset")

    dataset_dir = st.text_input(r'Example on Mac - /Users/amendez/MyDocuments/EULAS/ ; Example on Windows - C:\Users\amendez\MyDocuments\EULAS\\')
    if not dataset_dir:
        st.warning('Please upload file')
        st.stop()
    filenames = [ os.path.join(dataset_dir,i) for i in os.listdir(dataset_dir)]
    filenames = [i for i in filenames if os.path.isfile(i)]
    
    st.write(" ")
    lengths = [pd.read_csv(i).shape[0] for i in filenames]
    st.write("Upload Complete. {} files were uploaded".format(len(filenames)))
    d = {}
    k = [i.split("/")[-1] for i in filenames]
    for i in range(len(filenames)):
        n = filenames[i].split("/")[-1]
        d[n] = filenames[i]
    plt.bar([i for i in d.keys()],lengths)
    plt.xlabel("Dataset Uploaded")
    plt.ylabel("Number of Examples")
    st.pyplot()

    st.subheader("Explore examples in datasets")
    f_path = st.selectbox('Select Dataset to display',options=[i for i in d.keys()] )
    df = pd.read_csv(d[f_path])
    num_rows = st.selectbox('Select number of rows to display ',options=[5, 10, 50, 100, 500, 1000, 5000, df.shape[0]])
    st.table(df.iloc[:num_rows])



    st.header("Select button to retrain Machine Learning model with new data")
    if st.checkbox("Retrain Model"):

        st.write("Set Parameters")
        num_epochs = st.number_input("Number of Epochs",min_value=1,max_value=100,step=1)
        batch_size = st.selectbox('Select batch size',options=[2, 4, 8, 16, 32, 64, 128])
        destination_folder = st.text_input('Filepath to folder where you want trained model to be saved')
        if not destination_folder:
            st.warning('Please upload file')
            st.stop()
        if st.button("Train Model"):
            retrain_model_main(dataset_dir,destination_folder,num_epochs,batch_size)
            st.write("Trained!")

