import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")

MODEL_FILE = "student_model.pkl"
ENCODER_FILE = "encoders.pkl"

st.title("🎓 Student Performance Prediction")
st.write("Predict a student's **math score** using demographic and academic details.")

# -------------------------------
# Helper functions
# -------------------------------
def train_and_save_model(df):
    df = df.copy()

    required_cols = [
        'gender',
        'parental level of education',
        'lunch',
        'test preparation course',
        'reading score',
        'writing score',
        'math score'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing these columns: {missing_cols}")

    categorical_cols = [
        'gender',
        'parental level of education',
        'lunch',
        'test preparation course'
    ]

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[
        [
            'gender',
            'parental level of education',
            'lunch',
            'test preparation course',
            'reading score',
            'writing score'
        ]
    ]
    y = df['math score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(encoders, f)

    return model, encoders, mse, r2


def load_model_and_encoders():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_FILE, "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


# -------------------------------
# Upload / train section
# -------------------------------
st.subheader("1) Upload Dataset")
uploaded_file = st.file_uploader("Upload StudentsPerformance.csv", type=["csv"])

model = None
encoders = None
mse = None
r2 = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully.")
        st.write("Preview of dataset:")
        st.dataframe(df.head())

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, encoders, mse, r2 = train_and_save_model(df)

            st.success("Model trained and saved successfully.")
            st.write(f"**Mean Squared Error:** {mse:.2f}")
            st.write(f"**R² Score:** {r2:.4f}")

    except Exception as e:
        st.error(f"Error while processing dataset: {e}")

# If saved model exists, load it
if model is None and os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    try:
        model, encoders = load_model_and_encoders()
        st.info("Saved model loaded successfully.")
    except Exception as e:
        st.error(f"Could not load saved model: {e}")

# -------------------------------
# Prediction section
# -------------------------------
st.subheader("2) Predict Math Score")

if model is not None and encoders is not None:
    gender = st.selectbox("Gender", list(encoders['gender'].classes_))
    parent_edu = st.selectbox(
        "Parental Level of Education",
        list(encoders['parental level of education'].classes_)
    )
    lunch = st.selectbox("Lunch Type", list(encoders['lunch'].classes_))
    test_prep = st.selectbox(
        "Test Preparation Course",
        list(encoders['test preparation course'].classes_)
    )

    reading_score = st.slider("Reading Score", 0, 100, 70)
    writing_score = st.slider("Writing Score", 0, 100, 70)

    if st.button("Predict Math Score"):
        try:
            gender_val = encoders['gender'].transform([gender])[0]
            parent_edu_val = encoders['parental level of education'].transform([parent_edu])[0]
            lunch_val = encoders['lunch'].transform([lunch])[0]
            test_prep_val = encoders['test preparation course'].transform([test_prep])[0]

            input_data = np.array([[
                gender_val,
                parent_edu_val,
                lunch_val,
                test_prep_val,
                reading_score,
                writing_score
            ]])

            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))  # keep within score range

            st.success(f"Predicted Math Score: {prediction:.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.warning("Upload the dataset and train the model first.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn")