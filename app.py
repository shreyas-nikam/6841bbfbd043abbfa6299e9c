
import streamlit as st

st.set_page_config(page_title="Algorithmic Insurance Premium Calculator", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("Algorithmic Insurance Premium Calculator")
st.divider()

st.markdown("""
## Algorithmic Insurance Premium Calculator

This application calculates the insurance premium for algorithmic insurance contracts, focusing on medical malpractice lawsuits related to malignant tumor detection.  It allows users to interact with key parameters and visualize their impact on the calculated premium.

**Key Concepts:**

*   **Algorithmic Insurance:** Insurance contracts designed to protect against the financial risks associated with errors made by machine learning algorithms.
*   **Classification Threshold (τ):** A threshold used to classify data points as positive or negative based on the output of a machine learning model.
*   **Sensitivity:** The ability of a model to correctly identify positive cases.
*   **Specificity:** The ability of a model to correctly identify negative cases.

**How to Use:**

1.  Adjust the input parameters in the sidebar.
2.  Observe the calculated premium and sensitivity analysis.
3.  Understand the trade-offs between different parameters.
""")

# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Premium Calculator", "Documentation"])

if page == "Premium Calculator":
    from application_pages.premium_calculator import run_premium_calculator
    run_premium_calculator()
elif page == "Documentation":
    from application_pages.documentation import run_documentation
    run_documentation()
# Your code ends

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
