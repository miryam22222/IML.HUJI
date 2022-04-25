from matplotlib import ticker

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from matplotlib import pyplot as pt
from matplotlib.pyplot import figure
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

CURR_YEAR = 2022
BUILDING_YEAR = 1950
MINIMAL_SQFT = 1000
TRAIN_PROPORTION = 0.75


def load_data(filename: str):
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
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df = design_matrix(df)
    return df.drop('price', axis=1), df['price']


def design_matrix(df: pd.DataFrame):
    df['yr_renovated'] = df[['yr_renovated', 'yr_built']].max(axis=1)
    filters = ((df['bedrooms'] > 0) &
               (df['bedrooms'] <= 15) &
               (df['bathrooms'] > 0) &
               (df['sqft_living'] >= MINIMAL_SQFT) &
               (df['sqft_lot'] >= MINIMAL_SQFT) &
               (df['floors'] >= 0) &
               (df['waterfront'].isin([0, 1])) &
               (df['view'].isin(range(5))) &
               (df['condition'].isin(range(1, 6))) &
               (df['grade'].isin(range(1, 15))) &
               (df['sqft_above'] >= MINIMAL_SQFT) &
               (df['yr_renovated'] <= CURR_YEAR) &
               (df['yr_renovated'] >= BUILDING_YEAR) &
               (df['sqft_living15'] >= MINIMAL_SQFT) &
               (df['sqft_lot15'] >= MINIMAL_SQFT) &
               (df['zipcode'] >= 10000) &
               (df['zipcode'] <= 99999) &
               (df['bathrooms'] / df['bedrooms'] <= 1) &
               (df['sqft_living'] == df['sqft_above'] + df['sqft_basement']))
    if 'price' in df.columns:
        price_filter = (df['price'] > 0)
        filters = filters & price_filter
    df = df[filters]
    new_df = pd.get_dummies(df, prefix='zipcode', columns=['zipcode'])
    new_df = new_df.drop(
        ['id', 'date', 'sqft_living', 'yr_built', 'lat', 'long'],
        axis=1)
    return new_df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
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
    for feature in X:
        if not ('zipcode' in feature or feature in ['id', 'date', 'sqft_living', 'yr_built', 'zipcode', 'lat', 'long']):
            plot_pearson_correlation(X[feature], y, output_path, feature)


def plot_pearson_correlation(x, y, output_path, feature):
    fig, ax = pt.subplots()
    ax.scatter(x, y, linewidths=0.5)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    pt.plot(x, p(x), "b-")
    ax.set(title="Price as a function of %s\n With %f Pearson correlation" % (feature, pearson(x, y)), xlabel=feature.capitalize(), ylabel="Price")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    pt.savefig(f'{output_path}_{feature}.png')


def pearson(x, y):
    return np.cov(x, y)[0, 1] / (np.std(x) * np.std(y))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "../ANSWERS/ex2/pictures/price_to")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, TRAIN_PROPORTION)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_X = np.concatenate((np.ones((test_X.shape[0], 1)), test_X), axis=1)
    reg_model = LinearRegression(True)
    percentage = np.arange(10, 101, 1)
    mean_losses = []
    std_losses = []
    for p in percentage:
        losses = []
        for i in range(10):
            sample = train_X.sample(frac=(p / 100))
            y = train_y.reindex_like(sample)
            reg_model.fit(sample.to_numpy(), y.to_numpy())
            loss = reg_model.loss(test_X, test_y.to_numpy())
            losses.append(loss)
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    fig, ax = pt.subplots()
    ax.plot(percentage, mean_losses, marker='o', mfc='r', ms=1.5)
    ax.set(title="Mean Loss Under Increasing Training Percentage", xlabel="Persentage (p%)",
           ylabel="Mean MSE loss")
    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)
    ax.fill_between(percentage, (mean_losses - 2 * std_losses), (mean_losses + 2 * std_losses), color='r', alpha=0.1)
    pt.show()
    pt.savefig('../ANSWERS/ex2/q2.png')

