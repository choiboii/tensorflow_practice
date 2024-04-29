import tensorflow as tf
import functions

if __name__ == '__main__':
    # functions.check_version()
    #initialize model with dataset
    x_train, y_train, x_test, y_test = functions.create_dataset()

    model = functions.create_model()
    n = 1
    predictions = model(x_train[:n]).numpy()
    print("predictions: ", predictions)

    #turns the predictions into probabilities of occuring based on the predictions made
    probabilities = tf.nn.softmax(predictions).numpy()
    print("probabilities: ",probabilities)
    
    #notice how the total of the probabilities is 1 (to ~5-6 decimal point accuracy)
    total = 0
    for i in range(n): #need 2 for loops [[probabilities[0]], probabilities[1], ... etc.]
        for prob in probabilities[i]:
            total += prob
        print("total for probabilities[{}]: {}".format(i, total))
        total = 0

    #loss function: used to quantify the difference between predicted and real values
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    #the approximate loss should be -log(1/10) ~= 2.36, 
    #since there are ten entries in each prediction class
    loss = loss_fn(y_train[:n], predictions).numpy()
    print(loss)

    
