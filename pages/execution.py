import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Exécution",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Contenu de l'application
st.title("Bienvenue dans mon application Streamlit")
st.write("C'est une démonstration de l'utilisation de st.set_page_config.")
