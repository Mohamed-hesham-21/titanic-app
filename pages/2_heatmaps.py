import streamlit as st
from utils.figures import plot_heatmap_figures

filtered_df = st.session_state["filtered_df"]

st.header(r"🔥 Correlation Matrices ($r_{xy}$) & Conditional Probabilities")
st.markdown(
    r"Linear correlation structures and 2D contingency heatmaps across socioeconomic and demographic strata."
)

plot_heatmap_figures(filtered_df)