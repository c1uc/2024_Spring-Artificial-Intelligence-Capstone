import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


class ConvertToRGB(object):
    def __call__(self, x):
        x = transforms.ToPILImage()(x)
        x = x.convert("RGB")  # Convert to RGB format, ignoring alpha channel
        return x


def load_data(data_dir, batch_size=1):
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ConvertToRGB(),  # Convert to RGB format
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizing with 3 channels (RGB)
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return dataloader, dataset.classes


def initialize_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def imshow(ax, inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        ax.title.set_text(title)


def evaluate_model(model, dataloader, class_names, device):
    model.eval()
    fig, axs = plt.subplots(20, 10, figsize=(100, 100))
    i = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            row = (i + j) // 10
            col = (i + j) % 10
            ax = axs[row, col]
            ax.axis('off')
            imshow(ax, inputs.cpu().data[j],
                   title='Predicted: {}\nTrue: {}'.format(class_names[preds[j]], class_names[labels[j]]))
        i += inputs.size()[0]

    plt.tight_layout()
    plt.savefig('resnet_eval.png')


def main():
    # Set parameters
    data_dir = './dataset_rgb'
    batch_size = 4

    dataloader, class_names = load_data(data_dir, batch_size)
    print(len(dataloader.dataset))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(len(class_names))
    model = model.to(device)

    model.load_state_dict(torch.load('./models/resnet_rgb_train_test_lr0.01_epoch50.pth'))

    evaluate_model(model, dataloader, class_names, device)


if __name__ == "__main__":
    main()
