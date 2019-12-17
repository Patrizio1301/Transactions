import tensorflow as tf
import numpy as np
import sys
from src.main.visualizations import plot_training_info


def modelling(input_steps,
              total_transactions,
              total_weekday,
              total_day,
              transaction_embedding_length,
              weekday_embedding_length,
              day_embedding_length,
              dim_concat):
    input_transactions = tf.keras.layers.Input(shape=(input_steps,), dtype='float32', name="transactions_input")
    embedding_transactions = tf.keras.layers.Embedding(input_dim=total_transactions,
                                                       output_dim=transaction_embedding_length,
                                                       input_length=input_steps)(input_transactions)
    transactions = tf.keras.layers.LSTM(512,
                                        return_sequences=True,
                                        input_shape=(input_steps, transaction_embedding_length))(embedding_transactions)

    # Weekday embedding branch
    input_weekday = tf.keras.layers.Input(shape=(input_steps,), dtype='float32', name="weekday_input")
    embedding_weekday = tf.keras.layers.Embedding(input_dim=total_weekday,
                                                  output_dim=weekday_embedding_length,
                                                  input_length=input_steps)(input_weekday)
    weekday = tf.keras.layers.Reshape(target_shape=(input_steps, weekday_embedding_length))(embedding_weekday)

    # Day embedding branch
    input_day = tf.keras.layers.Input(shape=(input_steps,), dtype='float32', name="day_input")
    embedding_day = tf.keras.layers.Embedding(input_dim=total_day,
                                                  output_dim=day_embedding_length,
                                                  input_length=input_steps)(input_day)
    day = tf.keras.layers.Reshape(target_shape=(input_steps, day_embedding_length))(embedding_day)

    # "concat" mode can only merge layers with matching output shapes except for the concat axis.
    concat = tf.keras.layers.Concatenate(axis=2)([transactions, weekday, day])

    # Everything continues in a single branch
    lstmConcat = tf.keras.layers.LSTM(512,
                                     return_sequences=False,
                                     input_shape=(input_steps, dim_concat))(concat)

    #reshape_1 = Reshape((INPUT_ACTIONS, 2))(input_time)
    #merge embeddings (5 x 50) and times (5 x 1), to have 5 x 51

    dense_1 = tf.keras.layers.Dense(1024, activation='relu')(lstmConcat)
    drop_1 = tf.keras.layers.Dropout(0.8)(dense_1)
    dense_2 = tf.keras.layers.Dense(1024, activation='relu')(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.8)(dense_2)
    output_transaction = tf.keras.layers.Dense(total_transactions, activation='softmax', name="output_transaction")(drop_2)
    output_weekday = tf.keras.layers.Dense(total_weekday, activation='softmax',name="output_weekday")(drop_2)
    output_day = tf.keras.layers.Dense(total_day, activation='softmax', name="output_day")(drop_2)

    model = tf.keras.Model(inputs=[input_transactions, input_weekday, input_day],
                           outputs=[output_transaction, output_weekday, output_day])
    return model