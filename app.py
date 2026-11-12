from pathlib import Path
import streamlit as st
import pandas as pd
from utils.data import load_data

# --- AUTHOR / MADE BY CONFIGURATION ---
AUTHOR_NAME = "Mohamed Hesham"
AUTHOR_ROLE = "Data Science & AI Engineering"
GITHUB_URL = "https://github.com/Mohamed-hesham-21"
LINKEDIN_URL = "https://www.linkedin.com/in/mohamed-hesham-9215402a5/"

# 1. Page Configuration (Must be called once in the main entry script)
st.set_page_config(
    page_title="Titanic Scientific EDA",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Define Pages for Navigation
pages = {
    "Analytics & Visuals": [
        st.Page("pages/1_overview.py", title="Overview & Distributions", icon="📊", default=True),
        st.Page("pages/2_heatmaps.py", title="Heatmaps & Correlations", icon="🔥"),
        st.Page("pages/3_multidimensional.py", title="Multidimensional Analysis", icon="🔬"),
    ],
    "Theory & Synthesis": [
        st.Page("pages/4_methodology.py", title="Mathematical Formulation", icon="📐"),
        st.Page("pages/6_conclusion.py", title="Empirical Conclusion", icon="🎯"), # <-- Added here
        st.Page("pages/5_data_explorer.py", title="Data Explorer", icon="🗄️"),
    ],
}

pg = st.navigation(pages)

# 3. Load Dataset
df = load_data()
st.session_state["raw_df"] = df

# --- GLOBAL SIDEBAR FILTERS (Shared Across All Pages) ---
with st.sidebar:
    img_path = Path("utils/image.jpg")
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.header(r"🔍 Filter Sample Space ($\Omega$)")

    classes = ["First", "Second", "Third"]
    selected_classes = st.multiselect("Passenger Class ($C$)", options=classes, default=classes)

    genders = df["sex"].dropna().unique().tolist()
    selected_genders = st.multiselect("Gender ($G$)", options=genders, default=genders)

    ports = df["port"].dropna().unique().tolist()
    selected_ports = st.multiselect("Embarkation Port ($E$)", options=ports, default=ports)

    outcomes = df["survival"].dropna().unique().tolist()
    selected_outcomes = st.multiselect("Survival Status ($Y$)", options=outcomes, default=outcomes)

    st.divider()

    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider(
        r"Age Domain $A \in [a_{\min}, a_{\max}]$",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
    )
    include_missing_age = st.checkbox(r"Include missing ages ($A = \text{NaN}$)", value=True)

    min_fare, max_fare = float(df["fare"].min()), float(df["fare"].max())
    fare_range = st.slider(
        r"Fare Domain $F \in [f_{\min}, f_{\max}]$ ($)",
        min_value=min_fare,
        max_value=max_fare,
        value=(min_fare, max_fare),
    )

    # --- SIDEBAR "MADE BY" BADGE ---
    st.divider()
    st.markdown(
        f"""
        <div style="
            background-color: rgba(128, 128, 128, 0.1);
            padding: 14px;
            border-radius: 10px;
            border-left: 4px solid #00CC96;
            font-size: 0.9em;
        ">
            <p style="margin: 0; font-weight: 600;">👨‍💻 Made by {AUTHOR_NAME}</p>
            <p style="margin: 2px 0 8px 0; color: gray; font-size: 0.85em;">{AUTHOR_ROLE}</p>
            <a href="{GITHUB_URL}" target="_blank" style="text-decoration: none; margin-right: 12px;">🔗 GitHub</a>
            <a href="{LINKEDIN_URL}" target="_blank" style="text-decoration: none;">💼 LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- APPLY GLOBAL FILTERS ---
age_mask = df["age"].between(age_range[0], age_range[1])
if include_missing_age:
    age_mask = age_mask | df["age"].isna()

fare_mask = df["fare"].between(fare_range[0], fare_range[1]) | df["fare"].isna()

filtered_df = df[
    (df["class"].isin(selected_classes))
    & (df["sex"].isin(selected_genders))
    & (df["port"].isin(selected_ports) | df["port"].isna())
    & (df["survival"].isin(selected_outcomes))
    & age_mask
    & fare_mask
]

# Store in session_state so all pages can read it
st.session_state["filtered_df"] = filtered_df

# --- SHARED TOP HEADER & KPI METRICS ---
st.title("🚢 Titanic Empirical & Statistical Analysis")
st.markdown(
    r"Exploratory analysis of passenger survival probability $P(Y=1 \mid X)$ across demographic and socioeconomic covariates."
)

if filtered_df.empty:
    st.warning("⚠️ Empty subsample ($n = 0$). Please broaden your filter criteria in the sidebar.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
n_sub = filtered_df.shape[0]
n_tot = df.shape[0]

with c1:
    st.metric(
        r"Subsample Size ($n$)",
        f"{n_sub:,}",
        delta=f"{n_sub - n_tot} from N={n_tot}",
        delta_color="off",
    )
with c2:
    overall_surv = df["survived"].mean() * 100
    filt_surv = filtered_df["survived"].mean() * 100
    st.metric(
        r"Survival Rate ($\hat{p}$)",
        f"{filt_surv:.1f}%",
        delta=f"{filt_surv - overall_surv:+.1f}% vs population",
    )
with c3:
    mu_age = filtered_df["age"].mean()
    sd_age = filtered_df["age"].std()
    st.metric(
        r"Mean Age ($\mu_{\text{age}} \pm \sigma$)",
        f"{mu_age:.1f} yrs" if pd.notna(mu_age) else "N/A",
        delta=f"± {sd_age:.1f} yrs (SD)" if pd.notna(sd_age) else None,
        delta_color="off",
    )
with c4:
    mu_fare = filtered_df["fare"].mean()
    med_fare = filtered_df["fare"].median()
    st.metric(
        r"Mean Fare ($\mu_{\text{fare}}$)",
        f"${mu_fare:.2f}" if pd.notna(mu_fare) else "N/A",
        delta=f"Median: ${med_fare:.2f}" if pd.notna(med_fare) else None,
        delta_color="off",
    )
with c5:
    mu_fam = filtered_df["family_size"].mean()
    st.metric(
        r"Mean Family Size ($\bar{S}_{\text{fam}}$)",
        f"{mu_fam:.2f}",
    )

st.divider()

# 4. Run Selected Page
pg.run()

# --- SHARED MAIN PAGE FOOTER ---
st.divider()
st.markdown(
    f"""
    <div style="text-align: center; color: gray; font-size: 0.9em; padding: 10px 0;">
        Made with ❤️ using <b>Streamlit</b>, <b>Pandas</b> & <b>Plotly</b> by <b>{AUTHOR_NAME}</b> ({AUTHOR_ROLE})<br>
        <a href="{GITHUB_URL}" target="_blank" style="color: #00CC96; text-decoration: none;">GitHub</a> • 
        <a href="{LINKEDIN_URL}" target="_blank" style="color: #00CC96; text-decoration: none;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)