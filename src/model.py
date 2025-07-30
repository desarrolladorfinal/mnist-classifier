import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Capa oculta
        self.fc2 = nn.Linear(128, 10)       # Capa de salida (10 dígitos)

    def forward(self, x):
        x = x.view(-1, 28 * 28)   # Aplanar imagen (batch_size, 784)
        x = F.relu(self.fc1(x))   # Activación ReLU
        x = self.fc2(x)           # Salida sin softmax (usa CrossEntropyLoss que lo incluye)
        return x

if __name__ == "__main__":
    import torch
    model = MNISTClassifier()
    dummy_input = torch.randn(64, 1, 28, 28)  # Batch de 64 imágenes
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Esperado: (64, 10)
