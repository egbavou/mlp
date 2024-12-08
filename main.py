import streamlit as st

st.title(":blue[IMPLEMENTATION D\'UN RESEAUX DE NEURONE DENSE AVEC SKLEARN]")
st.header(':blue[I]')
st.subheader("👨🏾‍💻 P")
st.text("Ma première application web avec Streamlit ! ")

st.sidebar.header("Auteurs")

authors = [
    {"name": "Emmanuel GBAVOU", "role": "Data Scientist", "contribution": ""},
    {"name": "GBANGBOCHE Olabissi", "role": "Data Scientist", "contribution": ""},
]

for author in authors:
    st.sidebar.write("---")
    st.sidebar.subheader(author["name"])
    st.sidebar.write(f"**Rôle** : {author['role']}")
    st.sidebar.write(f"**Contribution** : {author['contribution']}")