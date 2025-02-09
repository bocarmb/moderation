import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax


# Charger le modèle et le tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4)

# Fonction de prédiction
def predict_comment(comments):
    inputs = tokenizer(comments, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = softmax(outputs.logits, dim=1).squeeze().tolist()
    classes = ["Toxique", "Neutre", "Insulte", "Menace"]
    result = {classes[i]: round(probabilities[i] * 100, 2) for i in range(len(classes))}
    return result

# Interface Streamlit
st.title("Modération Automatique des Commentaires")
st.write("Entrez un commentaire pour analyser son niveau de toxicité.")

user_input = st.text_area("Votre commentaire")
if st.button("Analyser"):
    if user_input.strip():
        prediction = predict_comment(user_input)
        st.write("### Résultats de l'analyse :")
        for label, score in prediction.items():
            st.write(f"**{label}** : {score}%")
    else:
        st.warning("Veuillez entrer un commentaire.")
