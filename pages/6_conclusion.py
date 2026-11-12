import streamlit as st
import pandas as pd

# Retrieve datasets from session state
filtered_df = st.session_state.get("filtered_df")
raw_df = st.session_state.get("raw_df")

if filtered_df is None or raw_df is None:
    st.warning("⚠️ Session data not found. Please reload from the main entry point (`app.py`).")
    st.stop()

st.header("🎯 Empirical Findings & Conclusion")
st.markdown(
    r"Synthesis of exploratory data analysis, conditional probabilities $P(Y=1 \mid X)$, and socioeconomic survival determinants."
)

# --- KEY STATISTICAL TAKEAWAYS ---
st.markdown("### 1. Key Statistical Takeaways")

col_c1, col_c2, col_c3 = st.columns(3)

# Compute subset vs population stats safely
n_sub = len(filtered_df)
n_tot = len(raw_df)
sub_surv = (filtered_df["survived"].mean() * 100) if n_sub > 0 else 0.0
tot_surv = (raw_df["survived"].mean() * 100) if n_tot > 0 else 0.0

with col_c1:
    st.metric(
        "Subsample Survival Rate",
        f"{sub_surv:.1f}%",
        delta=f"{sub_surv - tot_surv:+.1f}% vs population baseline",
    )

with col_c2:
    if "class" in filtered_df.columns and not filtered_df.empty:
        class_surv = filtered_df.groupby("class")["survived"].mean() * 100
        best_class = class_surv.idxmax() if not class_surv.empty else "N/A"
        best_rate = class_surv.max() if not class_surv.empty else 0.0
        st.metric(
            "Highest Surviving Class",
            f"{best_class}",
            delta=f"{best_rate:.1f}% survival rate",
        )
    else:
        st.metric("Highest Surviving Class", "N/A")

with col_c3:
    if "sex" in filtered_df.columns and not filtered_df.empty:
        sex_surv = filtered_df.groupby("sex")["survived"].mean() * 100
        best_sex = sex_surv.idxmax() if not sex_surv.empty else "N/A"
        sex_rate = sex_surv.max() if not sex_surv.empty else 0.0
        st.metric(
            "Highest Surviving Demographic",
            f"{str(best_sex).capitalize()}",
            delta=f"{sex_rate:.1f}% survival rate",
        )
    else:
        st.metric("Highest Surviving Demographic", "N/A")

st.divider()

# --- FORMAL SUMMARY & HYPOTHESIS SYNTHESIS ---
st.markdown("### 2. Socioeconomic & Demographic Determinants of Survival")

st.markdown(
    r"""
Based on the empirical distributions, correlation matrices, and conditional probabilities analyzed across the dashboard, several definitive conclusions emerge regarding passenger survival on the Titanic:

1. **Socioeconomic Stratification ($C \times F$):** 
   There is a strong positive correlation between ticket fare ($F$) and passenger class ($C$), and both correlate significantly with survival probability $P(Y=1)$. First-class passengers enjoyed preferential access to evacuation resources, resulting in substantially higher conditional survival rates compared to second and third-class passengers.
   
2. **The "Women and Children First" Protocol ($G \times A$):** 
   Stratification by gender ($G$) and age ($A$) highlights that female passengers and children (age $< 18$) experienced markedly higher survival probabilities across all socioeconomic classes, reflecting maritime evacuation protocols of the era.

3. **Family Size Dynamics ($S$):** 
   Moderate family sizes ($S \in [2, 4]$) exhibited higher survival likelihoods than solo travelers or extremely large family units ($S > 5$), where coordination bottlenecks significantly decreased individual survival odds.
"""
)

st.divider()

# --- MATHEMATICAL CONCLUSION / MODEL IMPLICATIONS ---
st.markdown("### 3. Predictive Modeling Implications")
st.markdown(
    r"To formalize these findings into a predictive classification task, the conditional probability of survival can be modeled via Logistic Regression:"
)

st.latex(r"P(Y = 1 \mid \mathbf{X}) = \sigma\left(\beta_0 + \beta_1 \cdot \text{Class} + \beta_2 \cdot \text{Gender} + \beta_3 \cdot \text{Age} + \beta_4 \cdot \text{Fare} + \beta_5 \cdot \text{FamilySize}\right)")

st.markdown(
    r"Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ represents the standard logistic sigmoid function. The exploratory data analysis confirms that features such as gender, passenger class, and fare serve as high-weight predictors in determining passenger fate."
)