from dataset import load_data
from model import CNN, ACNN
from train import train_model
from test import test_model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Main function
def main():
    # Load data
    data_flag = 'pathmnist'
    train_loader, val_loader, test_loader, num_classes = load_data(data_flag)

    # Initialize models
    model = CNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_2 = ACNN(num_classes)
    criterion_2 = nn.CrossEntropyLoss()
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.001)

    # Train models
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
    train_losses_2, val_losses_2 = train_model(model_2, train_loader, val_loader, criterion_2, optimizer_2, num_epochs=20)

    # Test models
    print("CNN:")
    test_model(model, test_loader)
    print("ACNN:")
    test_model(model_2, test_loader)

    # Plot losses
    plot_losses(train_losses, val_losses, train_losses_2, val_losses_2)


# Plot training and validation losses
def plot_losses(train_losses, val_losses, train_losses_2, val_losses_2):
    min_length = min(len(train_losses), len(val_losses), len(train_losses_2), len(val_losses_2))
    epochs = range(1, min_length + 1)
    plt.figure()
    plt.plot(epochs, train_losses[:min_length], label='CNN Training Loss')
    plt.plot(epochs, val_losses[:min_length], label='CNN Validation Loss')
    plt.plot(epochs, train_losses_2[:min_length], label='ACNN Training Loss')
    plt.plot(epochs, val_losses_2[:min_length], label='ACNN Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
