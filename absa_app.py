import streamlit as st
import pandas as pd
import spacy
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load DistilBERT
@st.cache_resource
def load_distilbert_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert_absa_model")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert_absa_model")
    model.eval()
    return tokenizer, model

tokenizer_distilbert, distilbert_model = load_distilbert_model()

# Load classical models
@st.cache_resource
def load_assets():
    nlp = spacy.load("en_core_web_sm")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    logreg_model = joblib.load("logreg_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    return nlp, vectorizer, logreg_model, svm_model, rf_model

nlp, vectorizer, logreg_model, svm_model, rf_model = load_assets()

# Evaluation data
@st.cache_data
def load_evaluation_data():
    y_test = joblib.load("y_test.pkl")
    y_pred_log = joblib.load("y_pred_log.pkl")
    y_pred_svm = joblib.load("y_pred_svm.pkl")
    y_pred_rf = joblib.load("y_pred_rf.pkl")
    test_labels = joblib.load("test_labels.pkl")
    pred_labels = joblib.load("pred_labels.pkl")
    return y_test, y_pred_log, y_pred_svm, y_pred_rf, test_labels, pred_labels

# Text dataset
@st.cache_data
def load_text_data():
    df = joblib.load("text_dataset.pkl")
    return df

# Aspect extraction
def extract_aspects(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Single Review ABSA",
    "Model Performance Evaluation",
    "Opinion Mining"
])

# --- PAGE 1: ABSA ---
if page == "Single Review ABSA":
    st.title("Aspect-Based Sentiment Analysis")
    input_text = st.text_area("Enter a review:")
    if input_text:
        aspects = extract_aspects(input_text)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        results = []

        for aspect in aspects:
            combined = f"{aspect} [SEP] {input_text}"
            inputs = tokenizer_distilbert(
                combined,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = distilbert_model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            results.append((aspect, label_map[pred]))

        df = pd.DataFrame(results, columns=["Aspect", "Predicted Sentiment"])
        st.table(df)

# --- PAGE 2: PERFORMANCE ---
elif page == "Model Performance Evaluation":
    st.title("Model Performance Evaluation")

    y_test, y_pred_log, y_pred_svm, y_pred_rf, test_labels, pred_labels = load_evaluation_data()

    results = {
        'Logistic Regression': (y_pred_log, y_test),
        'SVM': (y_pred_svm, y_test),
        'Random Forest': (y_pred_rf, y_test),
        'DistilBERT': (pred_labels, test_labels)
    }

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    model_names, accuracies, f1_scores = [], [], []

    for name, (preds, truth) in results.items():
        preds = np.array(preds)
        truth = np.array(truth)

        if isinstance(truth[0], str):  # If labels are strings
            truth = np.array([label_map[label.lower()] for label in truth])
            preds = np.array([label_map[label.lower()] for label in preds])

        model_names.append(name)
        accuracies.append(accuracy_score(truth, preds))
        f1_scores.append(f1_score(truth, preds, average='macro'))

    st.subheader("Model Accuracy & F1 Score Comparison")
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    st.pyplot(fig)

    st.subheader("Confusion Matrices")
    for name, (preds, truth) in results.items():
        preds = np.array(preds)
        truth = np.array(truth)

        if isinstance(truth[0], str):
            truth = np.array([label_map[label.lower()] for label in truth])
            preds = np.array([label_map[label.lower()] for label in preds])

        cm = confusion_matrix(truth, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])
        fig, ax = plt.subplots()
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        ax.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

# --- PAGE 3: OPINION MINING ---
elif page == "Opinion Mining":
    st.title("Opinion Mining")
    df = load_text_data()

    opinion_pairs = []
    for text in df['text'].dropna().sample(200, random_state=42):
        doc = nlp(text)
        for token in doc:
            if token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
                opinion_pairs.append({
                    'aspect': token.head.text,
                    'opinion': token.text,
                    'relation': 'amod',
                    'sentence': text
                })
            elif token.pos_ == 'VERB':
                for child in token.children:
                    if child.dep_ in ['dobj', 'nsubj'] and child.pos_ == 'NOUN':
                        opinion_pairs.append({
                            'aspect': child.text,
                            'opinion': token.text,
                            'relation': child.dep_,
                            'sentence': text
                        })

    opinions_df = pd.DataFrame(opinion_pairs)
    st.subheader("Extracted Aspect-Opinion Pairs")
    st.dataframe(opinions_df.head(20))

    st.subheader("Top Opinion Words")
    top_opinions = opinions_df['opinion'].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_opinions.values, y=top_opinions.index, ax=ax, palette="rocket")
    ax.set_title("Top 10 Most Common Opinion Words")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Opinion Word")
    st.pyplot(fig)

    st.subheader("WordCloud of Opinion Words")
    opinion_text = " ".join(opinions_df['opinion'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(opinion_text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("WordCloud of Opinion Words", fontsize=16)
    st.pyplot(fig)
