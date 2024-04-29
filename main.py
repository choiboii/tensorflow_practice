import tensorflow as tf
import functions

if __name__ == '__main__':
    # functions.check_version()
    #initialize model with dataset
    x_train, y_train, x_test, y_test = functions.create_dataset()

    model = functions.create_model()

    predictions = model(x_train[:1]).numpy()
    print(predictions)

    probabilities = tf.nn.softmax(predictions).numpy()
    print(probabilities)
