from dataset import load_data
from model import CNN, SE_CNN
from train import train_model
from test import test_model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


# Main function
def main():
    # Load data
    data_flag = 'pathmnist'
    train_loader, val_loader, test_loader, num_classes = load_data(data_flag)

    # Initialize models
    model = CNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model_2 = SE_CNN(num_classes)
    criterion_2 = nn.CrossEntropyLoss()
    optimizer_2 = optim.Adam(model_2.parameters(), lr=5e-4)

    # Train models
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
    if os.path.exists("best_model.pth"):
        if os.path.exists("CNN.pth"):
            os.remove("CNN.pth")
        os.rename("best_model.pth", "CNN.pth")
    train_losses_2, val_losses_2 = train_model(model_2, train_loader, val_loader, criterion_2, optimizer_2, num_epochs=20, patience=5)
    if os.path.exists("best_model.pth"):
        if os.path.exists("SE_CNN.pth"):
            os.remove("SE_CNN.pth")
        os.rename("best_model.pth", "SE_CNN.pth")

    # Test models
    print("CNN:")
    accuracy_1 = test_model(model, test_loader)
    print("SE_CNN:")
    accuracy_2 = test_model(model_2, test_loader)

    # Plot accuracy
    plot_accuracy(accuracy_1, accuracy_2)

    # Plot losses
    plot_losses(train_losses, val_losses, title='CNN')
    plot_losses(train_losses_2, val_losses_2, title='SE_CNN')


# Plot training and validation losses
def plot_losses(train_losses, val_losses, title=''):
    length = min(len(train_losses), len(val_losses))
    epochs = range(1, length + 1)
    plt.figure()
    plt.plot(epochs, train_losses[:length], label=title + 'Training Loss')
    plt.plot(epochs, val_losses[:length], label=title + 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title + 'Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_accuracy(accuracy_1, accuracy_2):
    data = [
        ["Model", "Accuracy"],
        ["CNN", f"{accuracy_1:.2f}"],
        ["SE_CNN", f"{accuracy_2:.2f}"]
    ]

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=None, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(data[0]))))
    plt.show()


if __name__ == "__main__":
    main()
