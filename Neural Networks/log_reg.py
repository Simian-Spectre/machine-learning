import numpy as np
import matplotlib.pyplot as plt
from csv_reader import np_arrays_from_csv, standardize_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_gradients(X, y, weights, v, est_type):
    N = X.shape[0]
    predictions = sigmoid(np.dot(X, weights))
    
    gradient_logistic = np.dot(X.T, (predictions - y)) / N
    gradient_logistic = np.ravel(gradient_logistic)
    
    if est_type == 'MAP':
        gradient_prior = weights / v
    elif est_type == 'ML':
        gradient_prior = 0
    else:
        print('ruh roh, bad estimate type!')
        return
    
    gradient = gradient_logistic + gradient_prior
    return gradient

def stochastic_gradient_descent(X_train, y_train, X_test, y_test, variances, gamma0, d, epochs, est_type):
    results = {}
    
    for v in variances:
        weights = np.zeros(X_train.shape[1])
        
        training_errors = []
        test_errors = []
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            learning_rate = gamma0 / (1 + (gamma0 / d) * epoch)

            for i in range(X_train_shuffled.shape[0]):
                X = X_train_shuffled[i:i+1]
                y = y_train_shuffled[i:i+1]

                gradient = compute_gradients(X, y, weights, v, est_type)

                weights -= learning_rate * gradient

            # Get training error
            train_predictions = sigmoid(np.dot(X_train, weights))
            train_error = np.mean((train_predictions - y_train) ** 2)
            training_errors.append(train_error)

            # Get eval error
            test_predictions = sigmoid(np.dot(X_test, weights))
            test_error = np.mean((test_predictions - y_test) ** 2)
            test_errors.append(test_error)

        results[v] = {
            "training_errors": training_errors,
            "test_errors": test_errors
        }

        # Plot!
        plt.plot(training_errors, label=f"Train v {v}")
        plt.plot(test_errors, label=f"Test v {v}")

    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.title("Training and Test Error Curves for Different Prior Variances")
    plt.show()

    return results

if __name__ == "__main__":
    # Loading data
    num_features = 4
    train_data_file = 'Perceptron\\Bank Note Data\\train.csv'
    eval_data_file = 'Perceptron\\Bank Note Data\\test.csv'
    train_features, train_labels = np_arrays_from_csv(train_data_file, num_features)
    eval_features, eval_labels = np_arrays_from_csv(eval_data_file, num_features)
    train_features = standardize_data(train_features)
    eval_features = standardize_data(eval_features)

    # Parameters
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    gamma0 = 0.001
    d = 100
    epochs = 100

    results = stochastic_gradient_descent(train_features, train_labels, eval_features, eval_labels, variances, gamma0, d, epochs, est_type='ML')

    # Prints
    for v, data in results.items():
        print(f"Variance: {v}")
        print(f"Final Training Error: {data['training_errors'][-1]}")
        print(f"Final Test Error: {data['test_errors'][-1]}\n")
