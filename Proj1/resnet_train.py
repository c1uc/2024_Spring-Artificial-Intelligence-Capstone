import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import os


def load_data(data_dir, batch_size=4):
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, len(dataset.classes)


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
    dataloader, num_classes = load_data(data_dir, batch_size)

    # Initialize model
    model = initialize_model(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    suffix = f'lr{learning_rate}_epoch{num_epochs}'
    writer = SummaryWriter(f'runs/resnet_rgb_{suffix}')

    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(dataloader, num_classes, model, criterion, optimizer, device, writer, num_epochs, suffix)

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
