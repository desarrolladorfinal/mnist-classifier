import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    """Retorna los dataloaders para entrenamiento y prueba."""

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convierte a tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normaliza con la media y desviación estándar de MNIST
    ])

    # Carga los datasets
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Crea los dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Prueba rápida
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

