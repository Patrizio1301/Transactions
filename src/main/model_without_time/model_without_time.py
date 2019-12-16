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

    dense_0 = tf.keras.layers.Dense(4, activation='relu')(embedding_actions)

    lstm_actions = tf.keras.layers.LSTM(units=512,
                                        return_sequences=False,
                                        input_shape=(INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH))(dense_0)

    #merge embeddings (5 x 50) and times (5 x 1), to have 5 x 51

    dense_1 = tf.keras.layers.Dense(512, activation='relu')(lstm_actions)
    drop_1 = tf.keras.layers.Dropout(0.8)(dense_1)
    dense_2 = tf.keras.layers.Dense(256, activation='relu')(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.8)(dense_2)
    output_actions = tf.keras.layers.Dense(total_actions, activation='softmax')(drop_2)

    model = tf.keras.Model(inputs=[input_transactions], outputs=[output_actions])

    # tf.keras.utils.plot_model(model, 'model.png')
    return model