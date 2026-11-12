import plotly.express as px
import streamlit as st
import pandas as pd
from utils.config import ORDERS, COLORS


def plot_overview_figures(df: pd.DataFrame):
    """Tab 1: Core distributions with mathematical captions."""
    col1, col2 = st.columns(2)

    with col1:
        outcome = df["survival"].value_counts().reset_index()
        outcome.columns = ["survival", "count"]
        fig_pie = px.pie(
            outcome,
            names="survival",
            values="count",
            hole=0.5,
            color="survival",
            color_discrete_map=COLORS,
            title="Bernoulli Outcome Proportions: P(Y = 1) vs. P(Y = 0)",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(r"Binary target distribution $Y \sim \text{Bernoulli}(p)$, where $p = P(\text{Survived})$.")

    with col2:
        fig_class = px.histogram(
            df,
            x="class",
            color="survival",
            barmode="group",
            category_orders=ORDERS,
            color_discrete_map=COLORS,
            labels={"class": "Socioeconomic Stratum (Class)", "survival": "Outcome (Y)"},
            title="Joint Frequency Distribution: f(Class, Y)",
        )
        fig_class.update_layout(yaxis_title="Frequency (n)")
        st.plotly_chart(fig_class, use_container_width=True)
        st.caption(r"Empirical frequency count $n_{c, y}$ across passenger class $c \in \{1, 2, 3\}$ and survival outcome $y \in \{0, 1\}$.")

    col3, col4 = st.columns(2)

    with col3:
        fig_age = px.histogram(
            df,
            x="age",
            color="survival",
            nbins=30,
            marginal="box",
            color_discrete_map=COLORS,
            labels={"age": "Age (Years)", "survival": "Outcome (Y)"},
            title="Age Distribution with Marginal Quartiles (IQR)",
        )
        fig_age.update_layout(yaxis_title="Density Count (n)")
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption(r"Top marginal boxplot displays median ($Q_2$), interquartile range ($\text{IQR} = Q_3 - Q_1$), and whiskers ($1.5 \times \text{IQR}$).")

    with col4:
        fig_fam = px.histogram(
            df,
            x="family_size",
            color="survival",
            nbins=11,
            barmode="group",
            color_discrete_map=COLORS,
            labels={"family_size": "Family Size (S = SibSp + Parch + 1)", "survival": "Outcome (Y)"},
            title="Discrete Distribution of Family Size (S)",
        )
        fig_fam.update_layout(bargap=0.1, yaxis_title="Frequency (n)")
        st.plotly_chart(fig_fam, use_container_width=True)
        st.caption(r"Discrete count distribution for $S_i = \text{SibSp}_i + \text{Parch}_i + 1$.")


def plot_heatmap_figures(df: pd.DataFrame):
    """Tab 2: Correlation matrix and conditional probability heatmaps."""
    with st.expander("📖 Mathematical Definitions for Heatmaps", expanded=False):
        st.latex(r"r_{xy} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}, \qquad \hat{P}(Y=1 \mid C=c, A=a) = \frac{1}{|\mathcal{I}_{c,a}|}\sum_{i \in \mathcal{I}_{c,a}} \mathbb{I}(Y_i = 1)")

    hm_col1, hm_col2 = st.columns(2)

    with hm_col1:
        plot_correlation_heatmap(df)

    with hm_col2:
        plot_survival_rate_heatmap(df)

    st.divider()
    plot_port_class_heatmap(df)


def plot_advanced_figures(df: pd.DataFrame):
    """Tab 3: Scatter plots and split violin KDE distributions."""
    col1, col2 = st.columns(2)

    with col1:
        clean_scatter = df.dropna(subset=["age", "fare"])
        fig_scatter = px.scatter(
            clean_scatter,
            x="age",
            y="fare",
            color="survival",
            size="family_size",
            hover_data=["name", "class", "sex"],
            color_discrete_map=COLORS,
            labels={"age": "Age (A)", "fare": "Fare Paid (F in $)", "survival": "Outcome (Y)"},
            title="Bivariate Joint Scatter: Fare (F) vs. Age (A) Scaled by Family Size (S)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(r"Each marker represents observation $(a_i, f_i)$ with radius proportional to $S_i = \text{SibSp}_i + \text{Parch}_i + 1$.")

    with col2:
        clean_violin = df.dropna(subset=["age", "class"])
        fig_violin = px.violin(
            clean_violin,
            x="class",
            y="age",
            color="survival",
            box=True,
            category_orders=ORDERS,
            color_discrete_map=COLORS,
            labels={"class": "Passenger Class (C)", "age": "Age (A)", "survival": "Outcome (Y)"},
            title="Conditional Kernel Density Estimation: f(Age | Class, Y)",
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        st.caption(r"Non-parametric Gaussian Kernel Density Estimate $\hat{f}_h(a) = \frac{1}{nh}\sum_{i=1}^n K\left(\frac{a - a_i}{h}\right)$ split by survival outcome.")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Pearson correlation matrix heatmap for numerical features."""
    num_cols = ["survived", "pclass", "age", "sibsp", "parch", "family_size", "fare"]
    available_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix (r ∈ [-1, 1])",
    )
    fig_corr.update_layout(coloraxis_colorbar=dict(title="r"))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(r"Pairwise Pearson correlation coefficients $r_{jk} \in [-1, 1]$. Note the negative correlation between `pclass` and `fare` ($r < -0.5$).")


def plot_survival_rate_heatmap(df: pd.DataFrame):
    """2D pivot heatmap of Conditional Survival Probability (%) by Class and Age Group."""
    pivot_df = (
        df.pivot_table(
            index="class",
            columns="age_group",
            values="survived",
            aggfunc="mean",
            observed=False,
        )
        * 100
    )

    class_order = [c for c in ["First", "Second", "Third"] if c in pivot_df.index]
    age_order = [
        a for a in ["child", "teen", "young adult", "adult", "senior"] if a in pivot_df.columns
    ]
    pivot_df = pivot_df.loc[class_order, age_order]

    fig_pivot = px.imshow(
        pivot_df,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        zmin=0,
        zmax=100,
        labels=dict(x="Age Stratum (A)", y="Class Stratum (C)", color="P(Y=1) %"),
        title="Conditional Survival Probability: P(Y = 1 | C, A) [%]",
    )
    st.plotly_chart(fig_pivot, use_container_width=True)
    st.caption(r"Empirical conditional probability $\hat{P}(Y=1 \mid C=c, A=a) \times 100\%$ across socioeconomic and age strata.")


def plot_port_class_heatmap(df: pd.DataFrame):
    """Contingency table heatmap across Embarkation Port and Class."""
    ct_df = pd.crosstab(df["port"], df["class"])
    class_order = [c for c in ["First", "Second", "Third"] if c in ct_df.columns]
    ct_df = ct_df[class_order]

    fig_ct = px.imshow(
        ct_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Passenger Class (C)", y="Embarkation Port (E)", color="Count (n)"),
        title="2D Contingency Matrix: Embarkation Port (E) × Class (C)",
    )
    st.plotly_chart(fig_ct, use_container_width=True)
    st.caption(r"Observed cell frequencies $O_{ij}$ in the $3 \times 3$ contingency table of Port $i$ by Class $j$.")


def plot_family_size(df: pd.DataFrame):
    """Backward-compatible wrapper."""
    plot_overview_figures(df)
    st.divider()
    plot_heatmap_figures(df)
    st.divider()
    plot_advanced_figures(df)