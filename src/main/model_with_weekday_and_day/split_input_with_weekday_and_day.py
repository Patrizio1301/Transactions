import numpy as np
import sys


def data_split(total_actions,
               X_transactions,
               X_times,
               y,
               y_weekday,
               y_day):
    #divide the examples in training and validation
    total_examples = len(X_transactions)
    test_per = 0.3
    limit = int(test_per * total_examples)
    x_actions_train = X_transactions[limit:]
    x_times_train = X_times[limit:]
    x_actions_test = X_transactions[:limit]
    x_times_test = X_times[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    y_weekday_train = y_weekday[limit:]
    y_weekday_test = y_weekday[:limit]
    y_day_train = y_day[limit:]
    y_day_test = y_day[:limit]
    print('Different actions:', total_actions)
    print('Total examples:', total_examples)
    print('Train examples:', len(x_actions_train), len(y_train))
    print('Test examples:', len(x_actions_test), len(y_test))
    sys.stdout.flush()
    x_train = np.array(x_actions_train)
    x_times_train = np.array(x_times_train)
    y_train = np.array(y_train)
    y_weekday_train = np.array(y_weekday_train)
    y_day_train = np.array(y_day_train)
    x_test = np.array(x_actions_test)
    x_times_test = np.array(x_times_test)
    y_test = np.array(y_test)
    y_weekday_test = np.array(y_weekday_test)
    y_day_test = np.array(y_day_test)
    print('Shape (X,y):')
    print(x_train.shape)
    print(x_times_train[:,:,1].shape)
    print(y_train.shape)
    return x_train, x_times_train, y_train, y_weekday_train, y_day_train, x_test, x_times_test, y_test, y_weekday_test, y_day_test