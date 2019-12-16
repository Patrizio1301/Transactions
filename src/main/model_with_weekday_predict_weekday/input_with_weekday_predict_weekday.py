from pandas import DatetimeIndex
import pandas as pd
import tensorflow as tf
import numpy as np


class Input(object):

    def __init__(self, input_steps):
        self.input_steps = input_steps

    def date_handling(self, date):
        date = pd.concat([date.reset_index(drop=True), pd.Series(DatetimeIndex(date).year).rename('year')], axis=1)
        date['month'] = DatetimeIndex(date['date']).month-1
        date['day'] = DatetimeIndex(date['date']).day-1
        date = self.weekday(date)
        del date['date']
        return date.to_numpy()

    @staticmethod
    def weekday(date):
        date['weekday'] = date['date'].dt.dayofweek
        return date

    @staticmethod
    def transaction_handling(df):
        words = df['transaction'].values.tolist()
        vocab_size = 4
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False)
        tokenizer.fit_on_texts(words)
        action_index = tokenizer.word_index
        #translate actions to indexes
        actions_by_index = np.array([action_index[transaction]-1 for transaction in words])
        # return actions_by_index.reshape([-1, 1])
        return actions_by_index

    def timeseries(self, dates, transactions, transactions_original):
        last_action = len(transactions) - 1
        X_transactions = []
        X_times = []
        y_transaction = self.output_onehot(transactions, transactions_original, values=['Alquiler', 'Metropolitan', 'Adeslas', 'Nomina'])
        y_weekday = self.output_onehot(transactions, transactions_original, values=range(7))
        for i in range(last_action-self.input_steps):
            X_transactions.append(transactions[i:i+self.input_steps])
            X_times.append(dates[i:i+self.input_steps])
        return X_transactions, X_times, y

    def output_onehot(self, transactions, transactions_original, values):
        last_action = len(transactions) - 1
        y = []
        for i in range(last_action-self.input_steps):
            target_action = transactions_original[i+self.input_steps]
            target_action_onehot = np.zeros(len(values))
            target_action_onehot[values.index(target_action)] = 1.0
            y.append(target_action_onehot)
        return y

    def __call__(self, df, *args, **kwargs):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        dates = self.date_handling(df["date"])
        transactions = self.transaction_handling(df)
        return self.timeseries(dates, transactions, df['transaction'].values.tolist())