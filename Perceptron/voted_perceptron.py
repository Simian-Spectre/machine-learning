from csv_reader import np_arrays_from_csv
import numpy as np


def voted_perceptron(examples, labels, num_epochs=10):
    num_examples = examples.shape[0]
    num_features = examples.shape[1]
    
    weight_vectors = []
    biases = []
    votes = []
    
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
                weights += (labels[i] - predicted_label) * examples[i]
                bias += (labels[i] - predicted_label)
                
                # Add weights, bias to this iteration, give it a vote
                weight_vectors.append(weights.copy())
                biases.append(bias)
                votes.append(1)  # One vote for this new model
                num_errors += 1
        
        # Converged!
        if num_errors == 0:
            print(f"Converged After {epoch + 1} Epochs")
            break

        # Reporting per epoch
        avg_error = num_errors / num_examples
        print(f"Epoch {epoch + 1}: Average Prediction Error {avg_error:.4f}")
    
    return weight_vectors, biases, votes


def predict(features, weight_vectors, biases, votes):
    # Weighted majority vote per example
    predictions = []
    num_features = features.shape[0]
    for i in range(num_features):
        weighted_sum = 0
        for j in range(len(weight_vectors)):
            prediction = 1 if np.dot(features[i], weight_vectors[j]) + biases[j] >= 0 else 0
            weighted_sum += votes[j] if prediction == 1 else -votes[j]
        
        # Prediction
        final_prediction = 1 if weighted_sum > 0 else 0
        predictions.append(final_prediction)
    
    return np.array(predictions)


def main():    
    num_features = 4
    # Loading data
    train_data_file = 'Perceptron\\Bank Note Data\\train.csv'
    eval_data_file = 'Perceptron\\Bank Note Data\\test.csv'
    train_features, train_labels = np_arrays_from_csv(train_data_file, num_features)
    eval_features, eval_labels = np_arrays_from_csv(eval_data_file, num_features)
    
    # Testing
    weights_vectors, biases, votes = voted_perceptron(train_features, train_labels)
    eval_predictions = predict(eval_features, weights_vectors, biases, votes)

    # Reporting trained voted perceptron
    for i in range(len(weights_vectors)):        
        print(f"Weight vectors {i}: {weights_vectors[i]}")

    # Reporting results
    num_correct_preds = 0
    num_examples = len(eval_labels)
    for i in range(num_examples):
        if eval_predictions[i] == eval_labels[i]:
            num_correct_preds += 1
    percent_correct = num_correct_preds/num_examples
    print(f"Accuracy: {percent_correct}")

if __name__ == "__main__":
    main()
