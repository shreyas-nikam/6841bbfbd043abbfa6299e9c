
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def run_premium_calculator():
    st.header("Algorithmic Insurance Premium Calculator")

    st.subheader("Input Parameters")
    litigation_cost_false_negative = st.number_input("Litigation Cost (False Negative) - L", min_value=0, value=500000)
    litigation_cost_false_positive = st.number_input("Litigation Cost (False Positive) - K", min_value=0, value=100000)
    num_patients = st.number_input("Number of Patients (N)", min_value=1, value=100)
    contract_price_upper_bound = st.number_input("Contract Price Upper Bound (Hp)", min_value=1000, value=50000)
    classification_threshold = st.slider("Classification Threshold (τ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    def generate_synthetic_data(num_samples=1000):
        data = {
            'radius_mean': np.random.rand(num_samples) * 20,
            'texture_mean': np.random.rand(num_samples) * 30,
            'perimeter_mean': np.random.rand(num_samples) * 150,
            'area_mean': np.random.rand(num_samples) * 1200,
            'smoothness_mean': np.random.rand(num_samples) * 0.2,
            'compactness_mean': np.random.rand(num_samples) * 0.4,
            'concavity_mean': np.random.rand(num_samples) * 1,
            'concave points_mean': np.random.rand(num_samples) * 0.2,
            'symmetry_mean': np.random.rand(num_samples) * 0.4,
            'fractal_dimension_mean': np.random.rand(num_samples) * 0.1,
            'malignant': np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
        }
        df = pd.DataFrame(data)
        return df

    df = generate_synthetic_data()
    st.subheader("Synthetic Data Sample")
    st.dataframe(df.head())

    X = df.drop('malignant', axis=1)
    y = df['malignant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    st.subheader("Model Training Complete")

    def calculate_premium(L, K, N, tau, model, X_test, y_test):
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= tau).astype(int)

        tp = np.sum((predictions == 1) & (y_test == 1))
        tn = np.sum((predictions == 0) & (y_test == 0))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))

        expected_loss = (fn / len(y_test)) * L + (fp / len(y_test)) * K
        total_expected_loss = N * expected_loss

        return total_expected_loss

    premium = calculate_premium(litigation_cost_false_negative, litigation_cost_false_positive, num_patients, classification_threshold, model, X_test, y_test)

    st.subheader("Calculated Premium")
    st.write(f"Estimated Premium: ${premium:,.2f}")

    st.subheader("Sensitivity Analysis")

    thresholds = np.arange(0.1, 1.0, 0.05)
    premiums = [calculate_premium(litigation_cost_false_negative, litigation_cost_false_positive, num_patients, t, model, X_test, y_test) for t in thresholds]

    fig, ax = plt.subplots()
    ax.plot(thresholds, premiums)
    ax.set_xlabel("Classification Threshold (τ)")
    ax.set_ylabel("Estimated Premium")
    ax.set_title("Premium vs. Classification Threshold")
    st.pyplot(fig)
