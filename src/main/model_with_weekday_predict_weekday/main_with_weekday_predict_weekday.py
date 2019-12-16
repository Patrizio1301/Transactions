import sys
from src.main.visualizations import Timeline
from src.main.model_with_weekday_predict_weekday.input_with_weekday_predict_weekday import Input
from src.main.training import training
from src.main.model_with_weekday_predict_weekday.model_with_weekday_predict_weekday import modelling
from src.main.dataset_loading import loading
from src.main.split_input import data_split

# transaction dataset
DIR = '../data/'
# Dataset with vectors but without the action timestamps
DATASET_CSV = 'C:/Users/patrizio.guagliardo/PycharmProjects/Transactions/src/data/transactions2.csv'

INPUT_STEPS = 1
BATCH_SIZE = 5
TRANSACTION_EMBEDDING_LENGTH = 4
WEEKDAY_EMBEDDING_LENGTH = 2
TOTAL_TRANSACTIONS = 4
TOTAL_WEEKDAYS = 7
EPOCHS = 100
learning_rate = 0.001


def main(path):
    df = loading(DATASET_CSV)
    print(df)
    Timeline(df)()
    x_transactions, x_times, y = Input(input_steps=INPUT_STEPS)(df)
    DIM_CONCAT = 6+15+1+3+TRANSACTION_EMBEDDING_LENGTH
    model = modelling(input_steps=INPUT_STEPS,
                      total_transactions=TOTAL_TRANSACTIONS,
                      total_weekday=TOTAL_WEEKDAYS,
                      transaction_embedding_length=TRANSACTION_EMBEDDING_LENGTH,
                      weekday_embedding_length=WEEKDAY_EMBEDDING_LENGTH,
                      dim_concat=DIM_CONCAT)

    x_train, x_times_train, y_train, x_test, x_times_test, y_test = data_split(TOTAL_TRANSACTIONS,
                                                                               x_transactions,
                                                                               x_times,
                                                                               y)

    input_train = [x_train, x_times_train[:,:,3], x_times_train[:,:,2], x_times_train[:,:,1], x_times_train[:,:,0]]
    output_train = y_train

    input_test = [x_test, x_times_test[:,:,3], x_times_test[:,:,2], x_times_test[:,:,1], x_times_test[:,:,0]]
    output_test = y_test

    training(model=model,
             input_steps=INPUT_STEPS,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             learning_rate=learning_rate,
             input_train=input_train,
             output_train=output_train,
             input_test=input_test,
             output_test=output_test)


if __name__ == "__main__":
    main(sys.argv)