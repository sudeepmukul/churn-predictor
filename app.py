import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("data.csv")

X = data[["Spend", "Logins", "Tickets", "Months", "Satisfaction"]]
y = data["Churn"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# UI
st.title("📊 Customer Churn Predictor")
st.write("Predict whether a customer is likely to leave.")

spend = st.number_input("Monthly Spend", 0, 10000, 1000)
logins = st.number_input("Monthly Logins", 0, 100, 5)
tickets = st.number_input("Support Tickets", 0, 20, 2)
months = st.number_input("Subscription Length (Months)", 0, 60, 6)
satisfaction = st.slider("Satisfaction Score", 1, 10, 5)

if st.button("Predict"):
    user_data = [[spend, logins, tickets, months, satisfaction]]
    prediction = model.predict(user_data)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
        st.write("Suggested Action: Offer discount / improve support / re-engage user.")
    else:
        st.success("✅ Customer likely to stay.")
        st.write("Suggested Action: Upsell premium plan / loyalty rewards.")