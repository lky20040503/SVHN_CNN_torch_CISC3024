from network import reconize_network
from Augmentation import augmentation_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torch import optim
from torchvision import datasets
import torch
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ToTensorV2(p=1.0)
])


def dump_model(model):
    with open('../model1.pkl', 'wb') as f:
        pickle.dump(model, f)


def load_model(model):
    with open('../model.pkl', 'rb') as f:
        model = pickle.load(f)


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, number_epoch=10):
    train_losses = []
    test_losses = []

    for epoch in range(number_epoch):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * len(images)

            test_losses.append(test_loss / len(test_loader))
            print(
                f'Epoch [{epoch + 1}/{number_epoch}], Train loss: {train_losses[-1]:.4f}, Test loss: {test_losses[-1]:.4f}')

    torch.save(model.state_dict(), "save.pt")

    return train_losses, test_losses


number_epoch = 50
model = reconize_network().to(device)
train_set = augmentation_dataset(root='./data', split='train', download=True, transform=transformer)
test_set = augmentation_dataset(root='./data', split='test', download=True, transform=transformer)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum=0.9, weight_decay=0.0005)

train_losses, test_losses = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer,number_epoch)

plt.figure(figsize=(5, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Curves")

