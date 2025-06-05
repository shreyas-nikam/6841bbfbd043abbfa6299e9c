
import streamlit as st

def run_documentation():
    st.header("Documentation")

    st.subheader("Important Definitions, Examples, and Formulae")

    st.markdown(
        """
        *   **Algorithmic Insurance:** Insurance contracts designed to protect against the financial risks associated with errors made by machine learning algorithms.

        *   **Conditional Value-at-Risk (CVaR):** A risk measure that quantifies the expected loss given that the loss exceeds a certain threshold. In the context of this application, CVaR estimates the potential financial loss an insurance company might face due to medical malpractice lawsuits arising from errors in tumor detection algorithms.

            *   Formula:
            ```latex
            CVaR_{\beta}(X) = E[X | X \geq VaR_{\beta}(X)]
            ```

            Where:
            *   `X` is the random variable representing the loss.
            *   `\beta` is the confidence level.
            *   `VaR_{\beta}(X)` is the Value-at-Risk at confidence level `\beta`.

            *Example:* A CVaR at 95% confidence estimates the expected loss, given that the loss is in the worst 5% of cases.

        *   **Value-at-Risk (VaR):** A risk measure that estimates the maximum loss expected over a specific time horizon at a given confidence level.
            *   Formula:

            ```latex
            P(X \leq VaR_{\beta}(X)) = \beta
            ```

            Where:
            *   `X` is the random variable representing the loss.
            *   `\beta` is the confidence level.

            *Example:* A VaR at 95% confidence level means that there is a 95% probability that the loss will not exceed the VaR value.

        *   **Classification Threshold (τ):** A threshold used to classify data points as positive or negative based on the output of a machine learning model. In this application, the threshold determines whether a patient is classified as having a malignant tumor.

            *Example:* If a model outputs a probability of 0.7 that a patient has a tumor, and the classification threshold is 0.5, the patient is classified as having a tumor.

        *   **Sensitivity:** The ability of a model to correctly identify positive cases (i.e., patients with tumors).
            *   Formula:

            ```latex
            Sensitivity = \frac{True Positives}{True Positives + False Negatives}
            ```

        *   **Specificity:** The ability of a model to correctly identify negative cases (i.e., patients without tumors).
            *   Formula:

            ```latex
            Specificity = \frac{True Negatives}{True Negatives + False Positives}
            ```
        *   **Equation 7 (from the research paper, representing the Claim Cost):**
        ```latex
        S = (1- \kappa_{\tau})K + (1 - \lambda_{\tau})L
        ```
        Where:
        * S is the claim cost
        * K is the litigation cost of a false positive
        * L is the litigation cost of a false negative
        * κ is the specificity
        * λ is the sensitivity
        """
    )

    st.subheader("Libraries and Tools")
    st.markdown(
        """
        *   **Streamlit:** Used for building the interactive user interface, handling user inputs, and displaying results and visualizations.
        *   **Pandas:** Used for data manipulation, creating dataframes for synthetic datasets, and potentially for loading data from external files.
        *   **NumPy:** Used for numerical computations, generating synthetic data, and performing calculations for premium estimation and sensitivity analysis.
        *   **Scikit-learn:** Used for training a machine learning model (e.g., Random Forest) to predict the probability of malignant tumors.
        *   **Matplotlib/Plotly:** Used for creating interactive charts to visualize the relationship between different parameters and the calculated premium. `matplotlib` is a static plotting library, while `plotly` allows for dynamic interactive plots.
        """
    )
