import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image


def load_images(base_folder):
    images = []
    labels = []
    label_encoder = LabelEncoder()

    for label_name in os.listdir(base_folder):
        label_folder = os.path.join(base_folder, label_name)
        if not os.path.isdir(label_folder):
            continue

        for filename in os.listdir(label_folder):
            img = Image.open(os.path.join(label_folder, filename))
            img = img.resize((180, 180))  # Resize images to 180x180
            img_data = np.array(img)
            images.append(img_data.flatten())  # Flatten the image into 1D array
            labels.append(label_name)

    images = np.array(images)
    labels = label_encoder.fit_transform(labels)

    return images, labels


def plot_images_clustered(images, labels, num_clusters):
    fig, axs = plt.subplots(num_clusters, 10, figsize=(10, num_clusters))
    for i in range(num_clusters):
        clustered_images = images[labels == i]
        for j in range(min(10, clustered_images.shape[0])):
            axs[i, j].imshow(clustered_images[j].reshape(180, 180, 3))
            axs[i, j].axis('off')
    plt.suptitle("Clustered Images")
    plt.tight_layout()
    plt.savefig("kmeans.png")
    plt.show()


def main():
    # Load images
    base_folder = './dataset_rgb'
    images, true_labels = load_images(base_folder)

    # Shuffle images
    images, true_labels = shuffle(images, true_labels, random_state=0)

    # Perform K-means clustering
    num_clusters = len(np.unique(true_labels))  # Assuming each folder corresponds to a different label
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    predicted_labels = kmeans.fit_predict(images)

    # Plot clustered images
    plot_images_clustered(images, predicted_labels, num_clusters)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
