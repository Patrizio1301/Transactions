import tensorflow as tf
import numpy as np
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
                                                    monitor='val_acc',
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
    plot_training_info(['acc', 'loss'], True, history.history)
    print('*' * 20)
    print('Evaluating best model...')
    sys.stdout.flush()
    metrics = model.evaluate(input_test, output_test, batch_size=batch_size)
    print(metrics)

    predictions = model.predict(input_test, batch_size)
    y_classes = predictions.argmax(axis=-1)
    correct = [0] * input_steps
    prediction_range = input_steps
    for i, prediction in enumerate(predictions):
        correct_answer = output_test[i].tolist().index(1)
        best_n = np.sort(prediction)[::-1][:prediction_range]
        for j in range(prediction_range):
            if prediction.tolist().index(best_n[j]) == correct_answer:
                for k in range(j, prediction_range):
                    correct[k] += 1

    accuracies = []
    for i in range(prediction_range):
        print('%s prediction accuracy: %s' % (i+1, (correct[i] * 1.0) / len(output_test)))
        accuracies.append((correct[i] * 1.0) / len(output_test))

    print(accuracies)

    print('************ FIN ************\n' * 3)