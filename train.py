from tqdm import tqdm
import torch
import sys
import os


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    model_save_path = 'best_model.pth'
    if os.path.exists(model_save_path):
        print(f"Deleting previous model file at {model_save_path}...\n")
        os.remove(model_save_path)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Add progress bar
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", file=sys.stdout, colour="red")

        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device).squeeze().long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Display current average loss on the progress bar
            train_progress.set_postfix({"Training Loss": f"{running_loss / (len(train_progress) + 1):.4f}"})

        train_losses.append(running_loss / len(train_loader))

        # Validate model
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)

        # Print validation results
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation Loss improved to {best_val_loss:.4f}. Model saved.\n")
        else:
            patience_counter += 1
            print(f"Validation Loss did not improve. Patience counter: {patience_counter}/{patience}\n")

        if patience_counter >= patience:
            print("Early stopping triggered.\n")
            break

    if os.path.exists(model_save_path):
        print("Loading the best model from checkpoint...\n")
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.to(device)
    return train_losses, val_losses


def validate_model(model, val_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    total, correct = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).squeeze().long()
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy
