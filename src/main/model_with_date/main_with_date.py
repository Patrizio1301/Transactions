import sys
from src.main.visualizations import Timeline
from src.main.input import Input
from src.main.training import training
from main.model_with_date.model_with_date import modelDesign
from src.main.dataset_loading import loading
from src.main.split_input import data_split

# transaction dataset
DIR = '../data/'
# Dataset with vectors but without the action timestamps
DATASET_CSV = 'C:/Users/patrizio.guagliardo/PycharmProjects/Transactions/src/data/transactions2.csv'

INPUT_STEPS = 1
BATCH_SIZE = 5
ACTION_EMBEDDING_LENGTH = 4
total_actions = 4
EPOCHS = 100
learning_rate=0.001


def main(path):
    df=loading(DATASET_CSV)
    print(df)
    Timeline(df)()
    x_transactions, x_times, y = Input(input_steps=INPUT_STEPS)(df)
    DIM_CONCAT = 6+15+1+3+ACTION_EMBEDDING_LENGTH
    model = modelDesign(INPUT_STEPS,
                        total_actions,
                        ACTION_EMBEDDING_LENGTH,
                        DIM_CONCAT)

    x_train, x_times_train, y_train, x_test, x_times_test, y_test = data_split(total_actions,
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