import streamlit as st
from utils.figures import plot_overview_figures

filtered_df = st.session_state["filtered_df"]

st.header("📊 Empirical Distributions & Marginal Densities")
st.markdown(
    r"Univariate and bivariate frequency distributions of survival outcome $Y$, passenger class $C$, age $A$, and family size $S$."
)

plot_overview_figures(filtered_df)