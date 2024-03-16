import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


def load_data(data_dir, batch_size=4):
    # train split transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # test split transforms
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    # split dataset into train and test
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(dataset.classes)


def train_model(dataloader, num_classes, model, criterion, optimizer, device, writer, num_epochs=10, suffix=''):
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Write loss and accuracy to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    print('Training complete')
    torch.save(model.state_dict(), f'./models/resnet_rgb_{suffix}.pth')
    return model


def evaluate_resnet(model, test_loader, writer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    writer.add_scalar('Accuracy/test', accuracy, 0)


def initialize_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def main(epoch=4, lr=0.001):
    # Set parameters
    data_dir = './dataset_rgb'
    batch_size = 4
    num_epochs = epoch
    learning_rate = lr

    # Load data
    train_loader, test_loader, num_classes = load_data(data_dir, batch_size)

    # Initialize model
    model = initialize_model(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    suffix = f'train_test_lr{learning_rate}_epoch{num_epochs}'
    writer = SummaryWriter(f'runs/resnet_rgb_train_test_{suffix}')

    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train_model(train_loader, num_classes, model, criterion, optimizer, device, writer, num_epochs, suffix)
    evaluate_resnet(model, test_loader, writer)

    # Close TensorBoard writer
    writer.close()


def grid_search():
    # Set parameters
    epochs = [10, 15, 20, 50]
    lrs = [0.001, 0.01, 0.1]

    for epoch in epochs:
        for lr in lrs:
            main(epoch, lr)


if __name__ == "__main__":
    grid_search()
