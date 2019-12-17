import sys
from src.main.visualizations import Timeline
from src.main.model_with_day_and_amount.input_with_day_and_amount import Input
from src.main.model_with_day_and_amount.training_with_day_and_amount import training
from src.main.model_with_day_and_amount.model_with_day_and_amount import modelling
from src.main.dataset_loading import loading
from src.main.model_with_day_and_amount.split_input_with_day_and_amount import data_split

# transaction dataset
DIR = '../data/'
# Dataset with vectors but without the action timestamps
DATASET_CSV = 'C:/Users/patrizio.guagliardo/PycharmProjects/Transactionsu/src/data/transactions2.csv'

INPUT_STEPS = 2
BATCH_SIZE = 5
TRANSACTION_EMBEDDING_LENGTH = 4
WEEKDAY_EMBEDDING_LENGTH = 2
DAY_EMBEDDING_LENGTH = 15
TOTAL_TRANSACTIONS = 4
TOTAL_WEEKDAYS = 7
TOTAL_DAYS=31
EPOCHS = 200
learning_rate = 0.001


def main(path):
    df = loading(DATASET_CSV)
    print(df)
    Timeline(df)()
    x_transactions, x_times, x_amount, y_transactions, y_weekday, y_day, y_amount = Input(input_steps=INPUT_STEPS)(df)
    DIM_CONCAT = 6+15+1+3+TRANSACTION_EMBEDDING_LENGTH
    model = modelling(input_steps=INPUT_STEPS,
                      total_transactions=TOTAL_TRANSACTIONS,
                      total_weekday=TOTAL_WEEKDAYS,
                      total_day=TOTAL_DAYS,
                      transaction_embedding_length=TRANSACTION_EMBEDDING_LENGTH,
                      weekday_embedding_length=WEEKDAY_EMBEDDING_LENGTH,
                      day_embedding_length=DAY_EMBEDDING_LENGTH,
                      dim_concat=DIM_CONCAT)

    x_train, x_times_train, x_amount_train, y_train, y_weekday_train, y_day_train, y_amount_train, x_test, x_times_test, x_amount_test, y_test, y_weekday_test, y_day_test, y_amount_test = data_split(TOTAL_TRANSACTIONS,
                                                                               x_transactions,
                                                                               x_times,
                                                                               x_amount,
                                                                               y_transactions,
                                                                               y_weekday,
                                                                               y_day,
                                                                               y_amount)

    input_train = [x_train, x_times_train[:,:,3], x_times_train[:,:,2], x_amount_train]
    output_train = [y_train, y_weekday_train, y_day_train, y_amount_train]

    input_test = [x_test, x_times_test[:,:,3], x_times_test[:,:,2], x_amount_test]
    output_test = [y_test, y_weekday_test, y_day_test, y_amount_test]

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