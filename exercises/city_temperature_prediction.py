import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
from matplotlib import pyplot as pt
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

MIN_TEMP = -15
MAX_TEMP = 40
MIN_DAY = 1
MAX_DAY = 31
MIN_MONTH = 1
MAX_MONTH = 12
CURR_YEAR = 2022


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
    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df = df.dropna()
    df = design_matrix(df)
    return df


def design_matrix(df: pd.DataFrame):
    filters = ((df['Temp'] >= MIN_TEMP) & (df['Temp'] <= MAX_TEMP))
    df.index = np.arange(len(df))
    dayOfYear = [date.dayofyear for date in df['Date']]
    df["DayOfYear"] = dayOfYear
    df = df[filters]
    return df


def scatter_temp_day(df: pd.DataFrame):
    x_arr = df['DayOfYear']
    y_arr = df['Temp']
    z_arr = df['Year']
    # pt.scatter(x_arr, y_arr, c=z_arr, s=3, cmap='RdYlBu')
    pt.scatter(x_arr, y_arr, c=z_arr, s=3)
    pt.xlabel("Day Of the Year")
    pt.xticks(np.arange(0, 365, 25))
    pt.ylabel("Temperature (c°)")
    pt.title("Temperature as a Function of the Day of the Year")
    pt.show()


def bar_std_month(df: pd.DataFrame):
    std = df.groupby(["Month", "DayOfYear"]).agg({'Temp': "std"})
    std = std.groupby("Month").mean().to_numpy()
    pt.bar(range(1, 13), std.flatten())
    pt.xticks(np.arange(1, 13))
    pt.xlabel("Month")
    pt.ylabel("Temperature STD")
    pt.title("STD of the Temperature as a Function of the Month")
    pt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    il_df = X.loc[X['Country'] == "Israel"]
    scatter_temp_day(il_df)
    bar_std_month(il_df)

    # Question 3 - Exploring differences between countries
    countries = X['Country'].unique()
    for country in countries:
        current_country = X.loc[X['Country'] == country]
        stds = current_country.groupby(
            ["Month", "DayOfYear"]).agg({'Temp': "std"}).groupby(
            "Month").mean().to_numpy()
        means = current_country.groupby(
            ["Month", "DayOfYear"]).agg({'Temp': "mean"}).groupby(
            "Month").mean().to_numpy()
        mean = means.reshape((1, 12))[0]
        std = stds.reshape((1, 12))[0]
        pt.plot(np.arange(1, 13), mean, label=country)
        pt.fill_between(np.arange(1, 13), mean + std, mean - std, alpha=0.1)
    pt.xticks(np.arange(1, 13))
    pt.legend(loc="upper left")
    pt.xlabel("Month")
    pt.ylabel("Mean of the Temperature")
    pt.title("Mean of the Temperature as a Function of the Month")
    pt.show()

    # Question 4 - Fitting model for different values of `k`
    il_temp_series = il_df.pop('Temp')
    il_days_df = pd.DataFrame({"DayOfYear": il_df.pop("DayOfYear")})
    train_X, train_y, test_X, test_y = split_train_test(il_days_df, il_temp_series, 0.75)
    losses = []
    degrees = [*range(1, 11)]
    for k in degrees:
        pol_model = PolynomialFitting(k)
        pol_model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = pol_model.loss(test_X.to_numpy(), test_y.to_numpy())
        losses.append(loss)
        print("Recorded a loss of: %f with a degree of %d" % (loss, k))
    pt.bar(degrees, losses)
    pt.xticks(np.arange(1, 11))
    pt.xlabel("Degree")
    pt.ylabel("Loss")
    pt.title("Loss as a Function of the Polynomial Model Degree")
    pt.show()

    # Question 5 - Evaluating fitted model on different countries
    min_k = np.argmin(losses)
    min_pol_model = PolynomialFitting(min_k)
    min_pol_model.fit(il_days_df.to_numpy(), il_temp_series.to_numpy())
    all_losses = []
    for country in countries:
        current_country = X.loc[X['Country'] == country]
        country_temp_series = current_country.pop('Temp')
        country_days_df = pd.DataFrame({"DayOfYear": current_country.pop("DayOfYear")})
        loss = min_pol_model.loss(country_days_df.to_numpy(), country_temp_series)
        all_losses.append(loss)
    pt.bar(countries, all_losses)
    pt.xlabel("Min K = %d" %(min_k))
    pt.ylabel("Loss")
    pt.title("Loss as a Function of Min K")
    pt.show()
