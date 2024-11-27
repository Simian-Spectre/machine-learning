import numpy as np
from csv_reader import np_arrays_from_csv

class DualSVM:
    def __init__(self, C=1.0, init_learning_rate=0.01, lr_func='a', k_type='s', sigma=1.0, max_epochs=100):
        self.C = C 
        self.init_learning_rate = init_learning_rate
        self.lr_func = lr_func
        self.max_epochs = max_epochs
        self.alpha = None
        self.X = None
        self.y = None
        self.k_type = k_type
        self.sigma = sigma
        self.K = None
        self.b = 0

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_examples, self.dimensions = X.shape
        self.alpha = np.zeros(self.num_examples)
        
        # Init the kernel
        if self.k_type == 's':
            self.K = standard_kernel(self.X)
        if self.k_type == 'G':
            self.K = gausian_kernel(self.X, self.sigma)

        init_learning_rate = self.init_learning_rate
        learning_rate = 0

        for epoch in range(self.max_epochs):
            if self.lr_func == 'a':
                learning_rate = calc_learning_rate_a(init_learning_rate, epoch)
            elif self.lr_func == 'b':
                learning_rate = calc_learning_rate_b(init_learning_rate, epoch)

            # Update each alpha
            for i in range(self.num_examples):
                decision_function = np.sum(self.alpha * self.y * self.K[i]) - self.b
                if self.y[i] * decision_function < 1:
                    # Misclassified
                    self.alpha[i] += learning_rate * (1 - self.y[i] * decision_function)
                else:
                    # Correctly classified
                    self.alpha[i] -= learning_rate * (1 - self.y[i] * decision_function)

                # Highly Motivated Bound Enforcement Officer >:|
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)

        self.w = np.dot(self.alpha * self.y, self.X)

    def predict(self, X):
        decision_function = np.sum(self.alpha * self.y * np.dot(X, self.X.T), axis=1) - self.b
        return np.sign(decision_function)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

def calc_learning_rate_a(learning_rate, epoch):
    return learning_rate / (1 + ((learning_rate / 3) * epoch))

def calc_learning_rate_b(learning_rate, epoch):
    return learning_rate / (1 + epoch)

def standard_kernel(X):
    return np.dot(X, X.T)

def gausian_kernel(X, sigma):
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-sq_dists / (2 * sigma**2))

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
        C_val = C / 873

        # Make and train standard Dual SVM
        dual_svm = DualSVM(C_val, init_learning_rate=0.01, lr_func='b', max_epochs=100)
        dual_svm.fit(train_features, train_labels)

        # Reporting
        train_accuracy = dual_svm.accuracy(train_features, train_labels)
        eval_accuracy = dual_svm.accuracy(eval_features, eval_labels)

        print(f"Train | S | {C_string}: {train_accuracy * 100:.2f}%")
        print(f"Test  | S | {C_string}: {eval_accuracy * 100:.2f}%")

        # Make and train Gaussian Dual SVM
        list_of_sigma_values = [0.1, 0.5, 1, 5, 100]
        for sigma in list_of_sigma_values:
            
            dual_svm = DualSVM(C=C_val, init_learning_rate=0.01, lr_func='b', k_type='G', sigma=sigma, max_epochs=100)
            dual_svm.fit(train_features, train_labels)

            # Reporting
            train_accuracy = dual_svm.accuracy(train_features, train_labels)
            eval_accuracy = dual_svm.accuracy(eval_features, eval_labels)

            print(f"Train | G | {C_string} | {sigma}: {train_accuracy * 100:.2f}%")
            print(f"Test  | G | {C_string} | {sigma}: {eval_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
