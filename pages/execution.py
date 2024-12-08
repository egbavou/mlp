import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

st.title(":blue[IMPLEMENTATION D\'UN RESEAUX DE NEURONE DENSE AVEC SKLEARN]")
st.header(':blue[Exécution du code avec SKLEARN]')

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

total_hidden_layers = st.number_input(
    'Combien de couches voulez vous crée dans votre réseau de neurones:',
    step=1, 
    format="%d",
    min_value=1,
    max_value=10
)

if total_hidden_layers :

    hidden_layers_array = [0] * total_hidden_layers

    for i in range(total_hidden_layers):
        hidden_layers_array[i] = st.number_input(
            f'Combien de neurones voulez vous sur la couche {i+1}:',
            step=1, 
            format="%d",
            min_value=1,
            max_value=500
        )

activation = st.selectbox('Choisissez votre fonction d\'activation ?:', ['relu', 'identity', 'logistic', 'tanh'])

solver = st.selectbox('Choisissez votre optimiseur ?:', ['lbfgs', 'sgd', 'adam'])

max_iter = st.number_input(
    f'Combien d\iteraction voulez vous au maximum ?:',
    step=1, 
    format="%d",
    min_value=20,
    max_value=500
)

hidden_layers_sizes = tuple(hidden_layers_array)

st.write(hidden_layers_sizes)


if st.button("Entrainer"):
    
    train_data = pd.read_csv('data/mnist_train.csv')
    test_data = pd.read_csv('data/mnist_test.csv')

    # Séparer les caractéristiques (pixels) et les étiquettes

    X_train = train_data.iloc[:, 1:].values  # Toutes les colonnes sauf la première
    y_train = train_data.iloc[:, 0].values   # La première colonne correspond aux étiquettes

    X_test = test_data.iloc[:, 1:].values  # Toutes les colonnes sauf la première
    y_test = test_data.iloc[:, 0].values   # La première colonne correspond aux étiquettes


    # Diviser les données en ensembles d'entraînement et de test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Étape 2 : Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Étape 3 : Définir le modèle avec des hyperparamètres
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers_sizes,  # Deux couches cachées avec 128 et 64 neurones
        activation=activation,            # Fonction d'activation : ReLU
        solver=solver,                # Optimiseur : Adam
        alpha=0.0001,                 # Régularisation L2
        learning_rate_init=0.001,     # Taux d'apprentissage
        max_iter=max_iter,                  # Nombre maximum d'époques
        random_state=42,
        verbose=True                  # Affiche les progrès pendant l'entraînement
    )

    # Étape 4 : Entraîner le modèle
    print("Entraînement du modèle...")
    model.fit(X_train, y_train)

    # Étape 5 : Évaluer les performances
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Étape 6 : Afficher la courbe de perte
    plt.plot(model.loss_curve_)
    plt.title('Courbe de perte')
    plt.xlabel('Itérations')
    plt.ylabel('Perte')
    st.pyplot(plt.show())
    # plt.show()

    # Visualiser quelques exemples avec prédictions
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for ax, img, true_label, pred_label in zip(axes.ravel(), X_test[:10], y_test[:10], y_pred[:10]):
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.set_title(f"V: {true_label}, P: {pred_label}")
        ax.axis('off')
    fig =plt.tight_layout()
    st.pyplot(fig)
    # plt.show()
