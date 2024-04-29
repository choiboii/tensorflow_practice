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

    #adam optimizer: short for "Adaptive Moment Estimation"
    #specializes in minimizing the loss function
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])  

    '''this is where machine learning actually happens:
        1. obtain data set (set of x_train, x_test, and y_train, y_test)
        2. fit model to the set of (x_train, y_train)
        3. evalulate to the actual values in dataset in (x_test, y_test)

        in step 2, we run multiple epochs, or trials, to increase accuracy of training
    '''
    #epochs == trials
    model.fit(x_train, y_train, epochs=5) 

    #98% accuracy after 5 trials!
    model.evaluate(x_test, y_test, verbose=2)