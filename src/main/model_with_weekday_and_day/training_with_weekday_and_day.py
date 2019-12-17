import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from src.main.visualizations import plot_training_info


def training(model,
             input_steps,
             batch_size,
             epochs,
             learning_rate,
             input_train,
             output_train,
             input_test,
             output_test):

    print('*' * 20)
    print('Building model...')
    sys.stdout.flush()

    adamOpti = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adamOpti,
                  metrics=['accuracy', 'mse', 'mae'])
    print(model.summary())
    sys.stdout.flush()

    print('*' * 20)
    print('Training model...')
    sys.stdout.flush()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                    monitor='val_accurracy',
                                                    verbose=0,
                                                    save_weights_only=False)

    history = model.fit(input_train,
                        output_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(input_test, output_test),
                        shuffle=False,
                        callbacks=[checkpoint])

    print('*' * 20)
    print('Plotting history...')
    sys.stdout.flush()
    plot_training_info(['output_transaction_acc', 'output_weekday_acc', 'loss'], True, history.history)
    print('*' * 20)
    print('Evaluating best model...')
    sys.stdout.flush()
    metrics = model.evaluate(input_test, output_test, batch_size=batch_size)
    print(metrics)

    predictions = model.predict(input_test, batch_size)
    y_transactions = output_test[0].argmax(axis=-1)
    y_weekday = output_test[1].argmax(axis=-1)
    y_day = output_test[2].argmax(axis=-1)

    y_transactions_pred = predictions[0].argmax(axis=-1)
    y_weekday_pred = predictions[1].argmax(axis=-1)
    y_day_pred = predictions[2].argmax(axis=-1)

    results = pd.DataFrame({'transaction': y_transactions, 'weekday': y_weekday, 'day': y_day,
                     'transaction_pred': y_transactions_pred, 'weekday_pred': y_weekday_pred, 'day_pred': y_day_pred})

    results.to_csv('results.csv')

    print('************ FIN ************\n' * 3)