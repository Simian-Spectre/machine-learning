from csv_reader import np_arrays_from_csv
import numpy as np

class PrimalSVM:
    def __init__(self, C=1.0, init_learning_rate=0.01, lr_func = 'a', max_epochs=100):
        self.C = C
        self.init_learning_rate = init_learning_rate
        self.lr_func = lr_func
        self.max_epochs = max_epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.num_examples, self.dimensions = X.shape
        self.w = np.zeros(self.dimensions)
        self.b = 0

        init_learning_rate = self.init_learning_rate
        learning_rate = 0

        for epoch in range(self.max_epochs):
            if(self.lr_func == 'a'):
                learning_rate = calc_learning_rate_a(init_learning_rate, epoch)
            if(self.lr_func == 'b'):
                learning_rate = calc_learning_rate_b(init_learning_rate, epoch)
            for i in range(self.num_examples):
                if y[i] * (np.dot(self.w, X[i]) + self.b) < 1:
                    # Misclassified
                    self.w -= learning_rate * (2 * self.C * self.w - np.dot(y[i], X[i]))
                    self.b -= learning_rate * (-y[i])
                else:
                    # Correctly classified
                    self.w -= learning_rate * (2 * self.C * self.w)

    def predict(self, X):
        decision_function = np.dot(X, self.w) + self.b
        return np.sign(decision_function)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
def calc_learning_rate_a(learning_rate, epoch):
        return learning_rate / (1 + ((learning_rate / 10) * epoch))
    
def calc_learning_rate_b(learning_rate, epoch):
        return learning_rate / (1 + epoch)

def main():
    # Loading data
    num_features = 4
    train_data_file = 'Perceptron\\Bank Note Data\\train.csv'
    eval_data_file = 'Perceptron\\Bank Note Data\\test.csv'
    train_features, train_labels = np_arrays_from_csv(train_data_file, num_features)
    eval_features, eval_labels = np_arrays_from_csv(eval_data_file, num_features)
    
    # Convert labels to {-1, 1}
    train_labels = 2 * train_labels - 1
    eval_labels = 2 * eval_labels - 1

    list_of_C_vals = [10, 100, 500, 700]
    for C in list_of_C_vals:
        C_string = f'{C}/873'
        C_val = C/873

        # Make and train SVM
        svm = PrimalSVM(C_val, init_learning_rate=0.01, lr_func='a', max_epochs=100)
        svm.fit(train_features, train_labels)

        # Reporting
        train_accuracy = svm.accuracy(train_features, train_labels)
        eval_accuracy = svm.accuracy(eval_features, eval_labels)

        print(f"Train | A | {C_string}: {train_accuracy * 100:.2f}%")
        print(f"Test  | A | {C_string}: {eval_accuracy * 100:.2f}%")

        # Make and train SVM
        svm = PrimalSVM(C=C_val, init_learning_rate=0.01, lr_func='b', max_epochs=100)
        svm.fit(train_features, train_labels)

        # Reporting
        train_accuracy = svm.accuracy(train_features, train_labels)
        eval_accuracy = svm.accuracy(eval_features, eval_labels)

        print(f"Train | B | {C_string}: {train_accuracy * 100:.2f}%")
        print(f"Test  | B | {C_string}: {eval_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
