import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_binary_dataset(file_path="binary_dataset.txt"):
    x, y = [], []
    with open(file_path, "r") as f:
        for line in f:
            binary_str, label = line.strip().split(",")
            x.append([int(bit) for bit in binary_str])
            y.append(int(label))
    return np.array(x), np.array(y)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    n = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true])
    loss = np.sum(log_likelihood) / n
    return loss

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

class CustomModel:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.z = np.dot(X, self.W) + self.b
        self.a = softmax(self.z)
        return self.a

    def backward(self, X, y_true, y_pred, lr=0.1):
        m = X.shape[0]
        dz = y_pred
        dz[range(m), y_true] -= 1
        dz /= m

        dW = np.dot(X.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        self.W -= lr * dW
        self.b -= lr * db

    def train(self, X, y, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y, y_pred)
            self.backward(X, y, y_pred, lr)
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

def plot_label_distribution(y_train, y_test):
    labels = np.arange(10)
    train_counts = [np.sum(y_train == i) for i in labels]
    test_counts = [np.sum(y_test == i) for i in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, train_counts, width, label='Train', color='skyblue')
    ax.bar(x + width/2, test_counts, width, label='Test', color='salmon')

    ax.set_xlabel('Digit Label')
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution in Train vs Test Set')
    ax.set_xticks(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

def train_and_evaluate():
    print("📥 Loading dataset...")
    X, y = load_binary_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    plot_label_distribution(y_train, y_test)

    input_dim = X_train.shape[1]
    output_dim = 10

    print("🧠 Initializing custom model...")
    model = CustomModel(input_dim, output_dim)

    print("🏋️‍♂️ Training...")
    model.train(X_train, y_train, epochs=1000, lr=0.5)

    print("📈 Evaluating...")
    y_pred = model.predict(X_test)

    print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

train_and_evaluate()
<<<<<<< HEAD
=======

>>>>>>> 308997de (added)
