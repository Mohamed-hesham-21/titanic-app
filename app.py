import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from utils.data import load_data
from utils.config import ORDERS, COLORS


df = load_data()

st.title('Titanic EDA')
st.dataframe(df, hide_index=True)
st.image('utils/image.jpg')
fig = px.histogram(df, x="family_size", nbins=11, labels={"family_size": "Family size (incl. the passenger)"},
                   title="How big were the families on board?")
fig.update_layout(bargap=0.1, yaxis_title="Passengers")
# fig.show()
st.plotly_chart(fig)


fig = px.histogram(df, x="age", nbins=30, marginal="box",
                   labels={"age": "Age"}, title="Age of the passengers")
st.plotly_chart(fig)

outcome = df["survival"].value_counts().reset_index()

fig = px.pie(outcome, names="survival", values="count", hole=0.5, color="survival",
             color_discrete_map=COLORS, title="Died vs survived")
st.plotly_chart(fig)