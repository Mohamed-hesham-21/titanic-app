import streamlit as st
import numpy as np

filtered_df = st.session_state["filtered_df"]
n_sub = len(filtered_df)

st.header("📐 Feature Engineering & Statistical Formulations")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("#### 1. Engineered Family Size ($S_i$)")
    st.markdown(
        r"For each passenger $i \in \{1, \dots, n\}$, total family size on board is defined as the sum of siblings/spouses ($\text{SibSp}_i$), parents/children ($\text{Parch}_i$), and the passenger themselves:"
    )
    st.latex(r"S_i = \text{SibSp}_i + \text{Parch}_i + 1")

    st.markdown("#### 2. Empirical Survival Estimator ($\hat{p}$)")
    st.markdown(
        r"Given binary survival indicator $Y_i \in \{0, 1\}$, the sample survival proportion and its 95% Wald confidence interval are:"
    )
    st.latex(
        r"\hat{p} = \frac{1}{n}\sum_{i=1}^{n} Y_i, \quad \text{CI}_{95\%} = \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}"
    )

    p_hat = filtered_df["survived"].mean()
    se_p = np.sqrt(p_hat * (1 - p_hat) / n_sub) if n_sub > 0 else 0
    ci_low = max(0.0, p_hat - 1.96 * se_p) * 100
    ci_high = min(1.0, p_hat + 1.96 * se_p) * 100
    st.info(
        f"**Live Subsample Estimate ($n={n_sub}$):** $\\hat{{p}} = {p_hat*100:.2f}\\%$ with $\\text{{SE}}(\\hat{{p}}) = {se_p*100:.2f}\\%$, yielding $\\text{{CI}}_{{95\\%}} = [{ci_low:.2f}\\%, {ci_high:.2f}\\%]$."
    )

with col_m2:
    st.markdown("#### 3. Pearson Product-Moment Correlation ($r_{xy}$)")
    st.markdown(
        r"Used in the correlation heatmap to quantify linear association between continuous/ordinal variables $X$ and $Y$:"
    )
    st.latex(
        r"r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}"
    )

    st.markdown("#### 4. Conditional Survival Rate per Stratum ($C \times A$)")
    st.markdown(
        r"In the 2D survival pivot heatmap, each cell $(c, a)$ represents the conditional expectation of survival given passenger class $C=c$ and age group $A=a$:"
    )
    st.latex(
        r"\hat{P}(Y = 1 \mid C = c, A = a) = \frac{\sum_{i \in \mathcal{I}_{c,a}} Y_i}{|\mathcal{I}_{c,a}|} \times 100\%"
    )