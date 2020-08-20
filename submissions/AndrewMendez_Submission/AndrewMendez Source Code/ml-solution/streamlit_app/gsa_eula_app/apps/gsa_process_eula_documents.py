import streamlit as st
import pandas as pd
import numpy as np

# For Loading BERT trained model
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from predict import *
from interpret import * 
import os
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
def app():
    
    # st.title("Home")
    st.title('Analyze End User License Agreement (EULA) Clauses using Interpretable Machine Learning')
    st.subheader("Developed by Andrew Mendez")
    

    @st.cache
    def get_pdf(filepath):
        pages_pdf = extract_clauses_from_pdf(filepath)
        clauses_pdf = preprocess_clauses_pdf(pages_pdf)
        return clauses_pdf
    @st.cache
    def get_docx(filepath):
        return get_text_from_docx(filepath)
    
    @st.cache
    def get_predictions(clauses):
        with st.spinner("Loading Model and predicting clause acceptabilty..."):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            PATH_TO_MODEL = '/Users/andrewmendez1/Documents/ai-ml-challenge-2020/data/Finetune BERT oversampling 8_16_2020/Model_1_4_0'
            best_model,tokenizer = load_model_and_tokenizer(PATH_TO_MODEL,device)
            ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
            sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
            cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
            print(device)

            acceptable_eulas = []
            unacceptable_eulas = []
            num_pos = 0
            for clause in clauses:
                pred,confidence = get_prediction_and_confidence(best_model,clause,tokenizer,device,ref_token_id,sep_token_id,cls_token_id)
                if int(pred)==0:
                    num_pos+=1
                    acceptable_eulas.append([clause,int(pred),float(confidence)])
                else:
                    unacceptable_eulas.append([clause,int(pred),float(confidence)])


        

        return acceptable_eulas,unacceptable_eulas, num_pos
    
    st.header("Welcome to the EULA Analyzer. This website allows you analyze EULA documents to determine whether terms and conditions are acceptable to the government.")
    st.write(" ")
    st.subheader("To get started, enter the file path for a PDF or Word document. Then, press Enter to upload.")

    filename = st.text_input(r'Example on Mac - /Users/amendez/MyDocuments/EULAS/testEULA.docx ; Example on Windows - C:\Users\amendez\MyDocuments\EULAS\testEULA.docx')
    if not filename:
        st.warning('Please upload file')
        st.stop()
    
    ext = os.path.splitext(filename)[1]
    name = filename.split("/")[-1]
    if ext == '.pdf':
        with st.spinner("Extracting clauses from .pdf..."):
            clauses = get_pdf(filename)
    elif ext == '.docx':
        with st.spinner("Extracting Clauses from .docx..."):
            clauses = get_docx(filename)

    st.write(" ")
    st.subheader("Upload is complete. The system has identified {} clauses.".format(len(clauses)))
    
    if st.checkbox("(Optional) Select to show all clauses extracted from the EULA"):
        st.subheader("Clauses from the {} EULA Document".format(name))
        index = st.slider('Click the slider and press left and right arrow keys to explore data.',0,len(clauses),1)
        st.write(HTML_WRAPPER.format(clauses[index]),unsafe_allow_html=True)

    st.subheader("Next, select checkbox to analyze EULA terms and conditions for acceptability.")
    
    if st.checkbox("Run Model"):
        st.subheader("Model Results")
        acceptable_eulas,unacceptable_eulas,num_pos = get_predictions(clauses)
        plt.bar(['Acceptable','Unacceptable'],np.array([num_pos,len(clauses)-num_pos]))
        plt.ylabel("Number of Clauses")
        plt.title("Overview of clauses predicted Acceptable/Unacceptable")
        plt.show()
        st.pyplot()
        st.write("The model has identified {} clauses as Acceptable, and {} clauses as Unacceptable.".format(num_pos,len(clauses)-num_pos))
        st.write(" ")
        st.subheader("Explore Acceptable clauses:")

        index1 = st.slider('Click the slider and press left and right arrow keys to explore data.',0,len(acceptable_eulas),1)
        label1 = ''
        if acceptable_eulas[index1][1] == 0:
            label1 = 'Acceptable'
        else:
            label1 = 'Unacceptable'
        st.subheader("The model has identified this clause as {} with {:.1f} % confidence.".format(label1,acceptable_eulas[index1][2]*100 ))
        st.write(HTML_WRAPPER.format(acceptable_eulas[index1][0]),unsafe_allow_html=True)
        st.write(" ")
        st.subheader("Explore Unacceptable clauses:")
        index2 = st.slider('Click the slider and press left and right arrow keys to explore data.',0,len(unacceptable_eulas),1)
        label2 = ''
        if unacceptable_eulas[index2][1] == 0:
            label2 = 'Acceptable'
        else:
            label2 = 'Unacceptable'
        st.subheader("The model has identified this clause as {} with {:.1f} % confidence.".format(label2,unacceptable_eulas[index2][2]*100 ))
        st.write(HTML_WRAPPER.format(unacceptable_eulas[index2][0]),unsafe_allow_html=True)
        st.subheader("Explore Model Interpretation")
        if st.checkbox("Select to see why individual clauses were identified as unacceptable."):
            # st.write(" Here we are leveraging the IntegratedGradients to interpret model predictions and show specific words that have highest attribution to the model output.")
            # st.write("Integrated gradients is an axiomatic attribution method that assigns an attribution(i.e. factor) score to each word/token in the clause.")
            # st.write(" To run, select the predicted unacceptable clause and press Interpret.")
            st.subheader("Explore why clauses were identified as Unacceptable:")
            index3 = st.slider('Click the slider and press left and right arrow keys to explore unacceptable.',0,len(unacceptable_eulas),1)
            st.write(HTML_WRAPPER.format(unacceptable_eulas[index3][0]),unsafe_allow_html=True)
            text = unacceptable_eulas[index3][0]
            if st.button("Interpret Prediction"):
                with st.spinner("Running Model Interpretation Analysis..."):
                    interpret_main(text,"?")
