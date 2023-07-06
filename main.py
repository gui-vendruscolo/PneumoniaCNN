import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DatasetConstructor(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []

        for label in os.listdir(root_dir):
            image_path = os.path.join(root_dir, label)
            self.image_paths.append(image_path)
            if "virus" in label:
                self.labels.append(1)
            elif "bacteria" in label:
                self.labels.append(2)
            else:
                self.labels.append(0)

        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index): 
        image_path = self.image_paths[index]
        label = self.labels[index]
        
        image = Image.open(image_path)  # Open image in grayscale mode
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
batch_size = 32
learning_rate = 0.001
num_epochs = 10
labels_strings = ["NORMAL", "PNEUMONIA VIRAL", "PNEUMONIA BACTERIANA"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset_normal = DatasetConstructor("./train/NORMAL", transform=transform)
train_dataset_pneumonia = DatasetConstructor("./train/PNEUMONIA", transform=transform)

val_dataset_pneumonia = DatasetConstructor("./val/PNEUMONIA/", transform=transform)
val_dataset_normal = DatasetConstructor("./val/NORMAL/", transform=transform)

test_dataset_pneumonia = DatasetConstructor("./test/PNEUMONIA/", transform=transform)
test_dataset_normal = DatasetConstructor("./test/NORMAL/", transform=transform)



train_dataset = torch.utils.data.ConcatDataset([train_dataset_normal, train_dataset_pneumonia])
val_dataset = torch.utils.data.ConcatDataset([val_dataset_normal, val_dataset_pneumonia])
test_dataset = torch.utils.data.ConcatDataset([test_dataset_normal, test_dataset_pneumonia])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

num_classes = 3  # Normal, Pneumonia Bacteriana, Pneumonia Viral



# Verificar se o arquivo de salvamento existe
if os.path.exists('modelo.pth'):
    # Carregar modelo salvo
    model = CNN(num_classes)
    model.load_state_dict(torch.load('modelo.pth'))
    model.eval()  # Definir o modelo para o modo de avaliação
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Modelo carregado.')

else:
    # Treinar o modelo
    print('Nenhum estado salvo encontrado. Iniciando o treinamento...')
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        correct = 0

        # Training
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
                
        train_loss = train_loss / len(train_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}")

# Validation
model.eval()
val_loss = 0.0
val_correct = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        val_loss += loss.item() * images.size(0)
        _, val_predicted = torch.max(outputs, 1)
        val_correct += (val_predicted == labels).sum().item()
        
    val_loss = val_loss / len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)

print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

 # Salvar o modelo treinado
torch.save(model.state_dict(), 'modelo.pth')
print('Modelo salvo.')

# Test
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_dataset)
accuracy = correct / total

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")


with torch.no_grad():
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # Converter tensores para NumPy arrays
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    # Exibir imagens em uma grade
    num_examples = 12  # Número de exemplos a serem mostrados
    num_rows = int(np.ceil(num_examples / 4))  # Número de linhas na grade
    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 2.5*num_rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_examples:
            image = np.transpose(images[i], (1, 2, 0))  # Converter de formato (C, H, W) para (H, W, C)
            image = np.squeeze(image)  # Remover dimensão desnecessária

            true_label = labels[i]
            predicted_label = predicted[i]

            ax.imshow(image, cmap='gray')
            ax.set_title(f'T: {labels_strings[true_label]}\nP: {labels_strings[predicted_label]}')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
