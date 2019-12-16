import sys
import pandas as pd

def loading(path):
    print('*' * 20)
    print('Loading dataset...')
    sys.stdout.flush()
    #dataset of activities
    DATASET = path
    df_dataset = pd.read_csv(DATASET, header=None)
    df_dataset.columns = ['date', 'weekday', 'transaction', 'amount']
    return df_dataset[df_dataset.transaction.notnull()]