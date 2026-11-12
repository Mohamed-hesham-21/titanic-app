import streamlit as st
from utils.figures import plot_advanced_figures

filtered_df = st.session_state["filtered_df"]

st.header("🔬 Joint Distributions & Kernel Density Estimation (KDE)")
st.markdown(
    r"Multivariate interactions between continuous ticket fare $F$, passenger age $A$, family size $S$, and class $C$."
)

plot_advanced_figures(filtered_df)