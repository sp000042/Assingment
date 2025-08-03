import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.title("🚢 Titanic Survival Prediction using Logistic Regression")

# Upload data
train_file = st.file_uploader("Upload Titanic Training CSV", type="csv")

if train_file:
    df = pd.read_csv(train_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    # Preprocessing
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df.drop(columns=['Cabin'], inplace=True, errors='ignore')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df.dropna(inplace=True)

    st.subheader("📈 Exploratory Data Analysis")
    fig1 = plt.figure()
    df.hist(figsize=(10, 8))
    st.pyplot(fig1)

    fig2 = plt.figure()
    sns.boxplot(data=df.select_dtypes(include='number'))
    st.pyplot(fig2)

    # Train model
    if 'Survived' in df.columns:
        X = df.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'], errors='ignore')
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📋 Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
