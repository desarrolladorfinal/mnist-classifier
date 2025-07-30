import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_data_loaders
from src.model import MNISTClassifier
import os

def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def main():
    # Configuraciones
    batch_size = 64
    learning_rate = 0.001
    epochs = 5
    model_path = "models/mnist_model.pth"
    os.makedirs("models", exist_ok=True)

    # Dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Datos
    train_loader, _ = get_data_loaders(batch_size=batch_size)

    # Modelo
    model = MNISTClassifier().to(device)

    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento
    train(model, train_loader, criterion, optimizer, device, epochs)

    # Guardar modelo
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en: {model_path}")

if __name__ == "__main__":
    main()

