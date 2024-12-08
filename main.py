import streamlit as st

st.title(":blue[IMPLEMENTATION D\'UN RESEAUX DE NEURONE DENSE AVEC SKLEARN]")
st.header(':blue[IMPLEMENTATION D\'UN RESEAUX DE NEURONE DENSE AVEC SKLEARN]')
st.subheader("👨🏾‍💻 P")
st.text("Ma première application web avec Streamlit ! ")

st.sidebar.header("Auteurs")
st.sidebar.text("Auteurs")
authors = [
    {"name": "Emmanuel GBAVOU", "role": "Data Scientist", "contribution": ""},
    {"name": "GBANGBOCHE Olabissi", "role": "Data Scientist", "contribution": ""},
]

for author in authors:
    st.subheader(author["name"])
    st.write(f"**Rôle** : {author['role']}")
    st.write(f"**Contribution** : {author['contribution']}")
    st.write("---")
    
texte_sidebar = st.sidebar.text_input("Entrez du texte","Abraham")
nombre_sidebar = st.sidebar.number_input("Entrez un nombre", min_value=0, max_value=100, value=27)

# Affichage des valeurs saisies dans le contenu principal
st.write(f"Vous avez saisi en barre latérale : Texte - **:blue[{texte_sidebar}]**, Nombre - **:blue[{nombre_sidebar}]**")


