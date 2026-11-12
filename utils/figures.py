import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.config import COLORS, LABELS, ORDERS


def create_family_size_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of passenger family sizes colored by survival."""
    fig = px.histogram(
        df,
        x="family_size",
        color="survival",
        nbins=11,
        barmode="group",
        color_discrete_map=COLORS,
        category_orders=ORDERS,
        labels=LABELS,
        title="Family Size Distribution by Outcome",
    )
    fig.update_layout(
        bargap=0.15,
        yaxis_title="Number of Passengers",
        legend_title_text="Outcome",
    )
    return fig


def create_age_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of passenger ages with a marginal box plot."""
    fig = px.histogram(
        df,
        x="age",
        color="survival",
        nbins=30,
        marginal="box",
        color_discrete_map=COLORS,
        category_orders=ORDERS,
        labels=LABELS,
        title="Age Distribution of Passengers",
    )
    fig.update_layout(
        yaxis_title="Number of Passengers",
        legend_title_text="Outcome",
    )
    return fig


def create_survival_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing overall survival proportions."""
    outcome = df["survival"].value_counts().reset_index()
    outcome.columns = ["survival", "count"]  # Safe across all pandas versions

    fig = px.pie(
        outcome,
        names="survival",
        values="count",
        hole=0.5,
        color="survival",
        color_discrete_map=COLORS,
        category_orders=ORDERS,
        labels=LABELS,
        title="Overall Survival Rate",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def create_class_survival_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing survival breakdown across passenger classes."""
    fig = px.histogram(
        df,
        x="class",
        color="survival",
        barmode="group",
        color_discrete_map=COLORS,
        category_orders=ORDERS,
        labels=LABELS,
        title="Survival by Passenger Class",
    )
    fig.update_layout(yaxis_title="Number of Passengers")
    return fig