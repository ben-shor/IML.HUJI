from pandas import DataFrame, Series
from plotly.subplots import make_subplots

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> Tuple[DataFrame, Series]:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]

    Info on columns
    -------
    sqft_living, the total house square footage of the house
    sqft_basement, size of the basement
    sqft_above = sqft_living - sqft_basement
    sqft_lot, lot size of the house
    sqft_living15, the average house square footage of the 15 closest houses
    sqft_lot15, the average lot square footage of the 15 closest houses
    """
    original_data = pd.read_csv(filename, index_col=0)

    # remove invalid rows (negative price or NaN in some column)
    processed_data = original_data.loc[~original_data.isnull().any(axis=1)]
    processed_data = processed_data.loc[processed_data["price"] > 0]

    # copy yr_built to yr_renovated if yr_renovated missing
    processed_data['yr_renovated'] = processed_data['yr_renovated'].mask(processed_data['yr_renovated'].eq(0),
                                                                         processed_data['yr_built'])

    processed_data["age_on_sale"] = pd.to_datetime(processed_data['date']).dt.year - processed_data['yr_built']
    processed_data["renovate_time_on_sale"] = pd.to_datetime(processed_data['date']).dt.year - \
                                              processed_data['yr_renovated']

    # diff from average living size
    processed_data["diff_from_sqft_lot15"] = processed_data["sqft_lot"] - processed_data["sqft_lot15"]
    processed_data["diff_from_sqft_living15"] = processed_data["sqft_living"] - processed_data["sqft_living15"]

    # one hot encoding for zipcode
    zipcode_dummies = pd.get_dummies(processed_data["zipcode"], prefix="zip")

    final_data = processed_data[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
                                 "view", "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15",
                                 "sqft_lot15", "age_on_sale", "renovate_time_on_sale", "diff_from_sqft_lot15",
                                 "diff_from_sqft_living15"]]
    final_data = pd.concat([final_data, zipcode_dummies], axis=1)
    return final_data, processed_data["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for count, col_name in enumerate(X):
        col_x = X[col_name]

        # calculate pearson correlation
        # based on https://stackabuse.com/calculating-pearson-correlation-coefficient-in-python-with-numpy/
        sigma_x = np.sqrt(np.mean(col_x**2) - np.mean(col_x)**2)
        sigma_y = np.sqrt(np.mean(y**2) - np.mean(y)**2)
        sigma_xy = np.mean(col_x * y) - np.mean(col_x) * np.mean(y)
        pearson = sigma_xy / (sigma_x * sigma_y)

        print(col_name, pearson)

        fig = go.Figure([go.Scatter(x=col_x, y=y, mode="markers")],
                        layout=go.Layout(title=f"{col_name} to price (Person={pearson})",
                                         xaxis={"title": col_name},
                                         yaxis={"title": "price"},
                                         height=400))
        fig.write_image(f"{output_path}/fig_{col_name}.png")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data_matrix, prices = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data_matrix, prices, output_path="../output")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_Y, test_X, test_Y = split_train_test(data_matrix, prices, train_proportion=0.25)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    ps = list(range(10, 101))
    losses = []
    for p in ps:
        linear_regressor = LinearRegression(include_intercept=True)

        p_losses = []
        for _ in range(10):
            p_train_X, p_train_Y, _, _ = split_train_test(train_X, train_Y, train_proportion=p / 100)
            linear_regressor.fit(p_train_X.to_numpy(), p_train_Y.to_numpy())
            p_losses.append(linear_regressor.loss(test_X.to_numpy(), test_Y.to_numpy()))
        losses.append(p_losses)
        # print(f"p={p} loss: {np.std(p_losses)} {p_losses}")
    fig = go.Figure([go.Scatter(x=ps,
                                y=[np.mean(l) for l in losses],
                                error_y=dict(
                                    type='data', # value of error bar given in data coordinates
                                    array=[2 * np.std(l) for l in losses],
                                    visible=True),
                                mode="markers")],
                    layout=go.Layout(title=f"percentage of train set used affect on loss",
                                     xaxis={"title": "p"},
                                     yaxis={"title": "loss"},
                                     height=1000))
    fig.show()
