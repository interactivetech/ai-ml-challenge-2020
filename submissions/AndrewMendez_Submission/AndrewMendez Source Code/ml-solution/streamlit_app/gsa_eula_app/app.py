import streamlit as st
from multiapp import MultiApp
from apps import gsa_process_eula_documents,retrain
app = MultiApp()

# apps here
app.add_app('Home',gsa_process_eula_documents.app)
app.add_app('Retrain',retrain.app)

#main app
app.run()