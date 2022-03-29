import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=[2]).drop_duplicates().dropna()
    df = df[df["Temp"] > -60]
    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_temps = df.loc["Israel"]
    fig = go.Figure([go.Scatter(x=israel_temps["DayOfYear"], y=israel_temps["Temp"], mode="markers",
                                marker=dict(color=israel_temps["Year"], colorscale="viridis"), showlegend=False)])
    fig.show()

    israel_by_month = israel_temps.groupby(['Month'], as_index=False).agg({'Temp':['mean','std'],'Year':'first'})
    fig = px.bar(x=israel_by_month["Month"], y=israel_by_month["Temp"]["std"], height=200)
    fig.show()

    # Question 3 - Exploring differences between countries
    df["Country_"] = df.index
    temp_by_month = df.groupby(['Month', 'Country_'], as_index=False).agg({'Temp': ['mean', 'std'], 'Year': 'first'})

    fig = go.Figure()
    for country in temp_by_month['Country_'].unique():
        country_temps = temp_by_month[temp_by_month['Country_'] == country]
        fig.add_traces([go.Scatter(x=country_temps["Month"],
                                   y=country_temps["Temp"]["mean"],
                                   error_y=dict(
                                       type='data', # value of error bar given in data coordinates
                                       array=country_temps["Temp"]["std"],
                                       visible=True),
                                   mode="markers",
                                   name=country
                                   )])
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_Y, test_X, test_Y = split_train_test(israel_temps["DayOfYear"],
                                                        israel_temps["Temp"],
                                                        train_proportion=0.25)
    losses = []
    for k in range(1, 11):
        poly_fitter = PolynomialFitting(k)
        poly_fitter.fit(train_X.to_numpy(), train_Y.to_numpy())
        losses.append(poly_fitter.loss(test_X.to_numpy(), test_Y.to_numpy()))
        print(k, round(losses[-1], 2))

    fig = px.bar(x=list(range(1, 11)), y=losses)
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    train_X, train_Y = israel_temps["DayOfYear"], israel_temps["Temp"]
    poly_fitter = PolynomialFitting(6)
    poly_fitter.fit(train_X.to_numpy(), train_Y.to_numpy())

    non_israel_temps = df[df["Country_"] != "Israel"]
    losses = []
    for country in non_israel_temps['Country_'].unique():
        country_temps = non_israel_temps[non_israel_temps['Country_'] == country]
        test_X, test_Y = country_temps["DayOfYear"], country_temps["Temp"]
        losses.append(poly_fitter.loss(test_X.to_numpy(), test_Y.to_numpy()))
        print(country, round(losses[-1], 2))

    fig = px.bar(x=non_israel_temps['Country_'].unique(), y=losses)
    fig.show()
