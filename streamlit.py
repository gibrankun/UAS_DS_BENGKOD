import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Streamlit App Title
st.title("Evaluasi Model Machine Learning")

# Data Upload
uploaded_file = st.file_uploader("Upload dataset Anda (CSV)", type="csv")
if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Input features dan target
    features = st.multiselect("Pilih fitur (X):", data.columns.tolist())
    target = st.selectbox("Pilih target (y):", data.columns.tolist())
    
    if features and target:
        X = data[features]
        y = data[target]

        # Pembagian data
        X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Normalisasi
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_original)
        X_test = scaler.transform(X_test_original)

        # Model
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "Naive Bayes": GaussianNB()
        }

        # Evaluasi Model
        st.write("## Hasil Evaluasi Model")
        for model_name, model in models.items():
            model.fit(X_train, y_train_original)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_original, y_pred)

            st.write(f"### Model: {model_name}")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test_original, y_pred, zero_division=0))

            # Confusion Matrix
            cm = confusion_matrix(y_test_original, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
