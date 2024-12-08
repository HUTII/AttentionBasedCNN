from dataset import load_data
from model import SE_CNN
from test import test_model_with_predictions, test_model
from sklearn.metrics import f1_score
import torch
import matplotlib.pyplot as plt


# Main function
def main():
    # Load data
    data_flag = 'pathmnist'
    _, _, test_loader, num_classes = load_data(data_flag)

    # Initialize model
    model = SE_CNN(num_classes)

    # Load model weights
    model_path = "SE_CNN.pth"
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    else:
        checkpoint = torch.load(model_path, weights_only=True)

    model.load_state_dict(checkpoint)
    print(f"Loaded model weights from {model_path}")

    # Test model and get predictions
    print("SE_CNN:")
    y_true, y_pred = test_model_with_predictions(model, test_loader)

    test_model(model, test_loader)

    # Calculate F1 Scores
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}")

    # Plot F1 Scores
    plot_f1_scores(micro_f1, macro_f1, weighted_f1)


# Plot F1 Scores
def plot_f1_scores(micro, macro, weighted):
    f1_scores = [micro, macro, weighted]
    labels = ['Micro-F1', 'Macro-F1', 'Weighted-F1']

    plt.figure()
    plt.bar(labels, f1_scores)
    plt.ylabel('F1 Score')
    plt.title('F1 Scores for SE_CNN Model')
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
