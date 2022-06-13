import pandas as pd

from DataCleaner import DataCleaner
from Forecaster import Forecaster


def read_data(path: str):
    return pd.read_csv(path)


if __name__ == '__main__':
    path_to_file = './data/m4_missing_values.csv'
    # read data
    my_example_data = read_data(path_to_file)

    # clean data
    data_cleaner = DataCleaner(my_example_data)
    clean_data = data_cleaner.trend_impute(8)

    # predict data
    forecast = Forecaster(clean_data, 500).predict_values()

    # save predictions
    forecast.to_csv('forecast.csv')
