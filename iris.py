import torch
import torch.nn as nn
import torch.optim as optim

from irisNet import IrisNet
from utils import load_data_from_csv

# Loading data and creating train and test sets
csv_path = "data/Iris.csv"
X_train, X_test, Y_train, Y_test = load_data_from_csv(csv_path)

# Creating an object of IrisNet class
model = IrisNet()

# Creating loss function
criterion = nn.CrossEntropyLoss()

# Creating optimizer
learning_rate = 0.15
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# Learning loop
epochs = 200

for epoch in range(epochs):
    # Turning model into training mode
    model.train()

    # Calculating model outputs
    outputs = model(X_train)

    # Calculating loss
    loss = criterion(outputs, Y_train)

    # Zeroing gradients
    optimizer.zero_grad()

    # Computing new gradients
    loss.backward()

    # Actualization of weights
    optimizer.step()

    # Printing loss every 20 epochs
    if(epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss {loss.item():.4f}")

# Turning model into evaluation mode
model.eval()

# Turning off gradient calculating
with torch.no_grad():

    # Model prediction for test set
    test_outputs = model(X_test)

    # Calculating model loss for test set
    test_loss = criterion(test_outputs, Y_test)

    # Conversion results to most predicted class
    _, predicted_classes = torch.max(test_outputs, 1)

    # Calculating prediction of model on test set
    correct_predictions = (predicted_classes == Y_test).sum().item()
    total_predictions = Y_test.size(0)
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")




