from pathlib import Path
import streamlit as st
from utils.data import load_data
from utils.figures import (
    create_age_distribution_chart,
    create_class_survival_chart,
    create_family_size_chart,
    create_survival_pie_chart,
)

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Titanic EDA Dashboard",
    page_icon="🚢",
    layout="wide",
)


def main() -> None:
    # 2. Load Data
    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # 3. Sidebar Filters & Banner Image
    with st.sidebar:
        # Check both assets/ and utils/ for backward compatibility
        img_path = Path("assets/image.jpg")
        if not img_path.exists():
            img_path = Path("utils/image.jpg")
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)

        st.header("Filter Passengers")

        selected_classes = st.multiselect(
            "Passenger Class",
            options=["First", "Second", "Third"],
            default=["First", "Second", "Third"],
        )

        selected_sex = st.multiselect(
            "Gender",
            options=df["sex"].dropna().unique().tolist(),
            default=df["sex"].dropna().unique().tolist(),
        )

        selected_ports = st.multiselect(
            "Embarkation Port",
            options=df["port"].dropna().unique().tolist(),
            default=df["port"].dropna().unique().tolist(),
        )

    # Apply filters
    filtered_df = df[
        (df["class"].isin(selected_classes))
        & (df["sex"].isin(selected_sex))
        & (df["port"].isin(selected_ports) | df["port"].isna())
    ]

    # 4. Header & KPI Metrics
    st.title("🚢 Titanic Exploratory Data Analysis")
    st.markdown("Interactive exploration of passenger demographics and survival factors.")

    total_passengers = len(filtered_df)
    survived_count = (filtered_df["survived"] == 1).sum()
    survival_rate = (survived_count / total_passengers * 100) if total_passengers > 0 else 0.0
    avg_age = filtered_df["age"].mean()
    avg_fare = filtered_df["fare"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", f"{total_passengers:,}")
    col2.metric("Survival Rate", f"{survival_rate:.1f}%", f"{survived_count:,} survived")
    col3.metric("Average Age", f"{avg_age:.1f} yrs" if not filtered_df["age"].isna().all() else "N/A")
    col4.metric("Average Fare", f"${avg_fare:.2f}" if not filtered_df["fare"].isna().all() else "N/A")

    st.divider()

    # 5. Main Content Tabs
    tab_charts, tab_data = st.tabs(["📊 Visual Analytics", "🗂️ Raw Dataset"])

    with tab_charts:
        if filtered_df.empty:
            st.warning("No passengers match the selected filter criteria.")
        else:
            # Row 1: Survival Pie & Class Breakdown
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.plotly_chart(
                    create_survival_pie_chart(filtered_df),
                    use_container_width=True,
                )
            with row1_col2:
                st.plotly_chart(
                    create_class_survival_chart(filtered_df),
                    use_container_width=True,
                )

            # Row 2: Age Distribution & Family Size
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.plotly_chart(
                    create_age_distribution_chart(filtered_df),
                    use_container_width=True,
                )
            with row2_col2:
                st.plotly_chart(
                    create_family_size_chart(filtered_df),
                    use_container_width=True,
                )

    with tab_data:
        st.subheader("Filtered Passenger Records")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()