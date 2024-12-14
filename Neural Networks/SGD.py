import numpy as np
import matplotlib.pyplot as plt
from csv_reader import np_arrays_from_csv

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output)
    output = sigmoid(output_input)

    return hidden_input, hidden_output, output_input, output

def back_propagation(X, y, weights_input_hidden, weights_hidden_output, hidden_input, hidden_output, output):
    # output error
    output_error = output - y
    output_delta = output_error * sigmoid_derivative(output)

    # hidden error
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    gradient_weights_hidden_output = np.dot(hidden_output.T, output_delta)
    gradient_weights_input_hidden = np.dot(X.T, hidden_delta)

    return gradient_weights_input_hidden, gradient_weights_hidden_output

def stochastic_gradient_descent(X_train, y_train, X_test, y_test, widths, gamma0, d, epochs, init_type):
    results = {}
    for width in widths:
        if init_type == 'random':
            # Initialize weights with Xavier initialization
            weights_input_hidden = np.random.randn(X_train.shape[1], width) * np.sqrt(1 / X_train.shape[1])
            weights_hidden_output = np.random.randn(width, 1) * np.sqrt(1 / width)
        elif init_type == 'zeros':
            # Initialize weights with zeros
            weights_input_hidden = np.zeros((X_train.shape[1], width))
            weights_hidden_output = np.zeros((width, 1))
        else:
            print('ruh roh, bad initialization type!')
            return

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

                hidden_input, hidden_output, output_input, output = forward_propagation(
                    X, weights_input_hidden, weights_hidden_output
                )

                gradient_weights_input_hidden, gradient_weights_hidden_output = back_propagation(
                    X, y, weights_input_hidden, weights_hidden_output, hidden_input, hidden_output, output
                )

                weights_input_hidden -= learning_rate * gradient_weights_input_hidden
                weights_hidden_output -= learning_rate * gradient_weights_hidden_output

            # Get training error
            _, _, _, train_output = forward_propagation(X_train, weights_input_hidden, weights_hidden_output)
            train_error = np.mean((train_output - y_train) ** 2)
            training_errors.append(train_error)

            # Get eval error
            _, _, _, test_output = forward_propagation(X_test, weights_input_hidden, weights_hidden_output)
            test_error = np.mean((test_output - y_test) ** 2)
            test_errors.append(test_error)

        results[width] = {
            "training_errors": training_errors,
            "test_errors": test_errors
        }

        # Plotting!
        plt.plot(training_errors, label=f"Train Width {width}")
        plt.plot(test_errors, label=f"Test Width {width}")

    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.title("Training and Test Error Curves")
    plt.show()

    return results


if __name__ == "__main__":
    # Loading data
    num_features = 4
    train_data_file = 'Perceptron\\Bank Note Data\\train.csv'
    eval_data_file = 'Perceptron\\Bank Note Data\\test.csv'
    train_features, train_labels = np_arrays_from_csv(train_data_file, num_features)
    eval_features, eval_labels = np_arrays_from_csv(eval_data_file, num_features)
    train_labels = train_labels.reshape(-1, 1)
    eval_labels = eval_labels.reshape(-1, 1)

    # Parameters
    widths = [5, 10, 25, 50, 100]
    gamma0 = 0.1
    d = 100
    epochs = 100

    results = stochastic_gradient_descent(train_features, train_labels, eval_features, eval_labels, widths, gamma0, d, epochs, 'random')

    # Prints
    for width, data in results.items():
        print(f"Width: {width}")
        print(f"Final Training Error: {data['training_errors'][-1]}")
        print(f"Final Test Error: {data['test_errors'][-1]}\n")
