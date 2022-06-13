import pandas as pd


class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data.iloc[2:]

    def forward_impute(self) -> pd.DataFrame:
        self.data['forward'] = self.data.iloc[:, 1].fillna(method='bfill')
        self.data.drop(self.data.columns[1], axis=1, inplace=True)
        return self.data

    def trend_impute(self, steps: int = 4) -> pd.DataFrame:
        rolling_mean = self.data.iloc[:, 1].rolling(steps, min_periods=1).mean()
        self.data['trend'] = self.data.iloc[:, 1].fillna(rolling_mean)
        self.data.drop(self.data.columns[1], axis=1, inplace=True)
        return self.data
