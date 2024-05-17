class DeepNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.weights1 = np.random.rand(input_size, hidden_size1) - 0.5
        self.bias1 = np.random.rand(1, hidden_size1) - 0.5
        self.weights2 = np.random.rand(hidden_size1, hidden_size2) - 0.5
        self.bias2 = np.random.rand(1, hidden_size2) - 0.5
        self.weights3 = np.random.rand(hidden_size2, output_size) - 0.5
        self.bias3 = np.random.rand(1, output_size) - 0.5
        self.learning_rate = learning_rate

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = sigmoid(self.z3)
        return self.a3

    def backward(self, X, y, output):
        # Backward pass
        output_error = y - output  # Error in output
        output_delta = output_error * sigmoid_derivative(output)

        a2_error = output_delta.dot(self.weights3.T)  # Error in second hidden layer
        a2_delta = a2_error * sigmoid_derivative(self.a2)

        a1_error = a2_delta.dot(self.weights2.T)  # Error in first hidden layer
        a1_delta = a1_error * sigmoid_derivative(self.a1)

        # Update weights and biases
        self.weights3 += self.learning_rate * self.a2.T.dot(output_delta)
        self.bias3 += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights2 += self.learning_rate * self.a1.T.dot(a2_delta)
        self.bias2 += self.learning_rate * np.sum(a2_delta, axis=0)
        self.weights1 += self.learning_rate * X.T.dot(a1_delta)
        self.bias1 += self.learning_rate * np.sum(a1_delta, axis=0)

    def fit(self, X, y, epochs=10000, batch_size=32):
        for _ in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == np.argmax(y, axis=1))
        return accuracy


# Assume X_train and y_train are preprocessed training data
# X_train: (number of samples, 784) -- flattened 28x28 images
# y_train: (number of samples, 10) -- one-hot encoded labels


import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Flatten the images
x_train = x_train.reshape(x_train.shape[0], -1)  # Shape (60000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)     # Shape (10000, 784)

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Initialize the network
nn = DeepNN(input_size=784, hidden_size1=3, hidden_size2=4, output_size=10, learning_rate=0.01)

# Train the network
nn.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predictions = nn.predict(X_test)

accuracy = nn.evaluate(X_test, y_test)
print(f'Test set accuracy: {accuracy * 100:.2f}%')