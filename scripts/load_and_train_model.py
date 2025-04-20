import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


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

def load_binary_dataset(file_path="binary_dataset.txt"):
    x, y = [], []
    with open(file_path, "r") as f:
        for line in f:
            binary_str, label = line.strip().split(",")
            x.append([int(bit) for bit in binary_str])
            y.append(int(label))
    return np.array(x), np.array(y)

def train_and_evaluate():
    print("Loading dataset...")
    x, y = load_binary_dataset()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    plot_label_distribution(y_train, y_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

train_and_evaluate()
