import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path

csv_path = Path("data/titanic.csv")
if not csv_path.exists():
    csv_path = Path("titanic_app/src/titanic_app/titanic.csv")
if not csv_path.exists():
    csv_path = Path("src/titanic_app/titanic.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()
df["class"] = df["pclass"].map({1: "First", 2: "Second", 3: "Third"})
df["survival"] = df["survived"].map({0: "Died", 1: "Survived"})
df["port"] = df["embarked"].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.", expand=False)
df["age_group"] = pd.cut(df["age"], [0, 12, 18, 35, 60, 80],
                         labels=["child", "teen", "young adult", "adult", "senior"])

ORDERS = {"class": ["First", "Second", "Third"], "survival": ["Died", "Survived"],
          "age_group": ["child", "teen", "young adult", "adult", "senior"]}
COLORS = {"Died": "#D95D4A", "Survived": "#0F766E"}
df.head(3)


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