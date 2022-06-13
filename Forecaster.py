import pandas as pd
from fbprophet import Prophet


class Forecaster:
    def __init__(self, clean_data: pd.DataFrame, training_size: int):
        self.model = None
        self.forecast = None
        self.clean_data = clean_data
        self.steps = training_size

    def get_train_data(self):
        return self.clean_data[-self.steps:]

    def get_test_data(self):
        return self.clean_data[:-self.steps]

    def get_model(self):
        train_data = self.get_train_data()
        train_data.columns = ['ds', 'y']
        train_data['ds'] = pd.to_datetime(train_data['ds'])

        model = Prophet()
        model.fit(train_data)
        self.model = model

    def predict_values(self):
        self.get_model()

        test_data = self.get_test_data()
        test_data.columns = ['ds', 'original']

        test_data['ds'] = pd.to_datetime(test_data['ds'])

        test_data.to_csv('test.csv')

        self.model.predict(test_data)
        self.forecast = self.model.predict(test_data)
        predictions = self.forecast['yhat']
        test_data['predictions'] = predictions
        return test_data
