import tensorflow as tf
import numpy as np
import sys
from src.main.visualizations import plot_training_info

def modelDesign(INPUT_ACTIONS,
          total_actions,
          ACTION_EMBEDDING_LENGTH,
          DIM_CONCAT):
    input_transactions = tf.keras.layers.Input(shape=(INPUT_ACTIONS,), dtype='float32', name="transactions_input")
    embedding_actions = tf.keras.layers.Embedding(input_dim=total_actions,
                                                  output_dim=ACTION_EMBEDDING_LENGTH,
                                                  input_length=INPUT_ACTIONS)(input_transactions)
    lstm_actions = tf.keras.layers.LSTM(512,
                                        return_sequences=True,
                                        input_shape=(INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH))(embedding_actions)

    # Weekday embedding branch
    input_weekday = tf.keras.layers.Input(shape=(INPUT_ACTIONS,), dtype='float32', name="weekday_input")
    embedding_weekday = tf.keras.layers.Embedding(input_dim=7,
                                                  output_dim=3,
                                                  input_length=INPUT_ACTIONS)(input_weekday)
    weekday = tf.keras.layers.Reshape(target_shape=(INPUT_ACTIONS, 3,))(embedding_weekday)

    # Day embedding branch
    input_day = tf.keras.layers.Input(shape=(INPUT_ACTIONS,), dtype='float32', name="day_input")
    embedding_day = tf.keras.layers.Embedding(input_dim=31,
                                              output_dim=15,
                                              input_length=INPUT_ACTIONS)(input_day)
    day = tf.keras.layers.Reshape(target_shape=(INPUT_ACTIONS, 15,))(embedding_day)

    # Month embedding branch
    input_month = tf.keras.layers.Input(shape=(INPUT_ACTIONS,), dtype='float32', name="month_input")
    embedding_month = tf.keras.layers.Embedding(input_dim=12,
                                                output_dim=6,
                                                input_length=INPUT_ACTIONS)(input_month)
    month = tf.keras.layers.Reshape(target_shape=(INPUT_ACTIONS, 6,))(embedding_month)

    # Month embedding branch
    input_year = tf.keras.layers.Input(shape=(INPUT_ACTIONS,), dtype='float32', name="year_input")
    year = tf.keras.layers.Reshape(target_shape=(INPUT_ACTIONS, 1))(input_year)

    # "concat" mode can only merge layers with matching output shapes except for the concat axis.
    concat = tf.keras.layers.Concatenate(axis=2)([lstm_actions, weekday, day, month, year])

    # Everything continues in a single branch
    lstm_both = tf.keras.layers.LSTM(512,
                                     return_sequences=False,
                                     input_shape=(INPUT_ACTIONS, DIM_CONCAT))(concat)

    #reshape_1 = Reshape((INPUT_ACTIONS, 2))(input_time)
    #merge embeddings (5 x 50) and times (5 x 1), to have 5 x 51

    dense_1 = tf.keras.layers.Dense(1024, activation='relu')(lstm_both)
    drop_1 = tf.keras.layers.Dropout(0.8)(dense_1)
    dense_2 = tf.keras.layers.Dense(1024, activation='relu')(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.8)(dense_2)
    output_actions = tf.keras.layers.Dense(total_actions, activation='softmax')(drop_2)

    model = tf.keras.Model(inputs=[input_transactions, input_weekday, input_day, input_month, input_year], outputs=[output_actions])

    # tf.keras.utils.plot_model(model, 'model.png')
    return model