from csv_reader import np_arrays_from_csv
import numpy as np


def perceptron(examples, labels, num_epochs=10):
    num_examples = examples.shape[0]
    num_features = examples.shape[1]

    # initialize zero bias and zeros for weights
    weights = np.zeros(num_features)
    bias = 0

    for epoch in range(num_epochs):
        num_errors = 0
        for i in range(num_examples):
            # Make prediction
            prediction = np.dot(examples[i], weights) + bias
            predicted_label = 1 if prediction >= 0 else 0

            # If an error is made:
            if predicted_label != labels[i]:
                weights += (labels[i] - (prediction >= 0)) * examples[i]
                bias += (labels[i] - (prediction >= 0))
                num_errors += 1

        # Converged!
        if num_errors == 0:
            print(f"Converged After {epoch + 1} Epochs")
            break

        # Reporting per epoch
        avg_error = num_errors / num_examples
        print(f"Epoch {epoch + 1}: Average Prediction Error {avg_error}")

    return weights, bias

def predict(features, weights, bias):
    return np.where(np.dot(features, weights) + bias >= 0, 1, 0)


def main():
    num_features = 4
    # Loading data
    train_data_file = 'Perceptron\\Bank Note Data\\train.csv'
    eval_data_file = 'Perceptron\\Bank Note Data\\test.csv'
    train_features, train_labels = np_arrays_from_csv(train_data_file, num_features)
    eval_features, eval_labels = np_arrays_from_csv(eval_data_file, num_features)
    
    weights, bias = perceptron(train_features, train_labels)
    
    # Reporting trained perceptron
    print("Final weights:", weights)
    print("Final bias:", bias)

    # Testing
    eval_predictions = predict(eval_features, weights, bias)
    num_correct_preds = 0
    num_examples = len(eval_labels)
    for i in range(num_examples):
        if eval_predictions[i] == eval_labels[i]:
            num_correct_preds += 1
    percent_correct = num_correct_preds/num_examples
    print(f"Accuracy: {percent_correct}")

if __name__ == "__main__":
    main()
