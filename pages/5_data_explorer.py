import streamlit as st

filtered_df = st.session_state["filtered_df"]

col_header, col_dl = st.columns([4, 1])
with col_header:
    st.header(f"🗄️ Filtered Passenger Records ($n = {len(filtered_df)}$)")
with col_dl:
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_bytes,
        file_name="titanic_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.dataframe(filtered_df, hide_index=True, use_container_width=True)

with st.expander("📈 Parametric & Non-Parametric Summary Moments", expanded=True):
    st.dataframe(filtered_df.describe(include="all").T, use_container_width=True)