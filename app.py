import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE ----------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

# ---------------- LOAD + TRAIN ----------------
@st.cache_resource
def load_model():
    # Read dataset
    data = pd.read_csv("customer_churn_dataset-training-master.csv")

    # Remove ID column
    if "CustomerID" in data.columns:
        data = data.drop("CustomerID", axis=1)

    # Fill missing values
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

    # Convert text to numbers
    encoders = {}
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Inputs and output
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simple + good model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Check accuracy
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return model, encoders, acc

model, encoders, acc = load_model()

# ---------------- UI ----------------
st.title("Customer Churn Predictor")
st.write("ML Project using Random Forest")
st.write(f"Accuracy: {acc:.2f}")

age = st.slider("Age", 18, 80, 30)
tenure = st.slider("Tenure", 1, 60, 12)
usage = st.slider("Usage Frequency", 1, 30, 10)
support = st.slider("Support Calls", 0, 10, 2)
delay = st.slider("Payment Delay", 0, 30, 5)
spend = st.number_input("Total Spend", 0, 5000, 500)
last = st.slider("Last Interaction", 1, 30, 5)

gender = st.selectbox("Gender", ["Male", "Female"])
subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

# ---------------- PREDICT ----------------
if st.button("Predict"):
    user_data = pd.DataFrame([{
        "Age": age,
        "Gender": encoders["Gender"].transform([gender])[0],
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support,
        "Payment Delay": delay,
        "Subscription Type": encoders["Subscription Type"].transform([subscription])[0],
        "Contract Length": encoders["Contract Length"].transform([contract])[0],
        "Total Spend": spend,
        "Last Interaction": last
    }])

    result = model.predict(user_data)[0]
    prob = model.predict_proba(user_data)[0][1]

    if result == 1:
        st.error(f"Likely to Leave ({prob:.2%})")
    else:
        st.success(f"Likely to Stay ({1-prob:.2%})")