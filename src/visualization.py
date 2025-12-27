import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

def plot_distribution(df, column, title=None, color='teal'):
    """Plots histogram and KDE for a column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=30, color=color)
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(column)
    plt.show()

def plot_correlation_heatmap(df, title='Correlation Matrix'):
    """Plots correlation heatmap for numerical columns."""
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

def plot_scatter_trend(df, x_col, y_col, color_col=None, title=None):
    """Plots interactive scatter plot with trendline."""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     trendline="ols" if color_col else None,
                     title=title or f'{x_col} vs {y_col}')
    fig.show()

def plot_temporal_trend(df, date_col, value_col, agg_func='mean', title=None):
    """Plots temporal trend."""
    trend = df.groupby(date_col)[value_col].agg(agg_func).reset_index()
    fig = px.line(trend, x=date_col, y=value_col, 
                  markers=True, title=title or f'Trend of {value_col} over {date_col}')
    fig.show()

def plot_choropleth(df, loc_col, val_col, year, title=None):
    """Plots world map choropleth."""
    df_year = df[df['Year'] == year]
    fig = px.choropleth(df_year, 
                        locations=loc_col, 
                        locationmode='country names',
                        color=val_col,
                        hover_name=loc_col,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title=title or f"{val_col} in {year}")
    fig.show()
