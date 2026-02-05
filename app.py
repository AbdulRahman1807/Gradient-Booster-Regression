import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered"
)

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            background-color: #0b0f14;
            color: #e6e6e6;
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            color: #7aa2f7;
            font-weight: 600;
        }

        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        div[data-baseweb="input"] > div {
            background-color: #0f1623;
            border: 1px solid #2a3b6e;
            border-radius: 10px;
            padding: 6px;
        }

        div[data-baseweb="input"] input {
            color: #e6e6e6;
        }

        div[data-baseweb="input"]:focus-within {
            border-color: #7aa2f7;
            box-shadow: 0 0 12px rgba(122, 162, 247, 0.35);
        }

        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #1f2a44, #2a3b6e);
            border: 1px solid #7aa2f7;
            color: #e6e6e6;
            padding: 0.75rem;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }

        .stButton > button:hover {
            box-shadow: 0 0 18px rgba(122, 162, 247, 0.45);
            transform: translateY(-1px);
        }

        .prediction-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background-color: #0f1623;
            border-radius: 14px;
            border: 1px solid #2a3b6e;
            box-shadow: 0 0 22px rgba(122, 162, 247, 0.25);
            font-size: 20px;
            text-align: center;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Diabetes Prediction App")
st.write(
    "Predicts the diabetes outcome value using a Gradient Boosting Regression model."
)

@st.cache_resource
def load_model():
    model = joblib.load("diabetes_gradient_boosting_model.pkl")
    feature_columns = joblib.load("diabetes_feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

st.subheader("Patient Data")

col1, col2 = st.columns(2)
input_data = {}

for i, feature in enumerate(feature_columns):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            step=0.1
        )

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted Outcome Value<br><br>
            <strong>{prediction:.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Gradient Boosting Regressor · Streamlit Application")
