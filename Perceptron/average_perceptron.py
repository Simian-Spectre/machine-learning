from csv_reader import np_arrays_from_csv
import numpy as np

def average_perceptron(features, labels, num_epochs=10):
    num_examples = features.shape[0]
    num_features = features.shape[1]
    
    # Initialize zero bias and zeros for weights
    weights = np.zeros(num_features)
    bias = 0

    # Keep track of total weights and bias
    total_weights = np.zeros(num_features)
    total_bias = 0

    for epoch in range(num_epochs):
        num_errors = 0
        for i in range(num_examples):
            # Make prediction
            prediction = np.dot(features[i], weights) + bias
            predicted_label = 1 if prediction >= 0 else 0
            
            # If an error is made:
            if predicted_label != labels[i]:
                weights += (labels[i] - predicted_label) * features[i]
                bias += (labels[i] - predicted_label)
                num_errors += 1

            total_weights += weights
            total_bias += bias
        
        # Reporting per epoch
        avg_error = num_errors / num_examples
        print(f"Epoch {epoch + 1}: Average Prediction Error {avg_error:.4f}")

        # Converged!
        if num_errors == 0:
            print(f"Converged After {epoch + 1} Epochs")
            break

    # Averages
    avg_weights = total_weights / num_examples
    avg_bias = total_bias / num_examples

    return avg_weights, avg_bias


def predict(features, avg_weights, avg_bias):
    return np.where(np.dot(features, avg_weights) + avg_bias >= 0, 1, 0)


def main():
    num_features = 4
    # Loading data
    train_data_file = 'Perceptron\\Bank Note Data\\train.csv'
    eval_data_file = 'Perceptron\\Bank Note Data\\test.csv'
    train_features, train_labels = np_arrays_from_csv(train_data_file, num_features)
    eval_features, eval_labels = np_arrays_from_csv(eval_data_file, num_features)
    
    weights, bias = average_perceptron(train_features, train_labels)
    
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