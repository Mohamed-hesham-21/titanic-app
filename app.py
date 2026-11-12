import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from utils.data import load_data
from utils.figures import plot_family_size


df = load_data()

st.title('Titanic EDA')
st.dataframe(df, hide_index=True)
st.image('utils/image.jpg')

plot_family_size(df)