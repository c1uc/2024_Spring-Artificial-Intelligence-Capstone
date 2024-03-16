import os
import numpy as np
from skimage import io, color, feature, transform
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib.pyplot as plt
from random import randint


def load_dataset(dataset_dir):
    images = []
    labels = []
    label_names = sorted(os.listdir(dataset_dir))
    for label, label_name in enumerate(label_names):
        label_dir = os.path.join(dataset_dir, label_name)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            image = io.imread(image_path)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels), label_names


def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray_image = color.rgb2gray(image)
        resized_image = transform.resize(gray_image, (128, 128))
        hog_feature = feature.hog(resized_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(hog_feature)
    return np.array(hog_features)


def visualize_classification(images, labels, label_names, predicted_labels):
    num_samples = 5
    num_classes = len(label_names)

    fig, axs = plt.subplots(num_classes, num_samples, figsize=(12, 12))

    for i, label_name in enumerate(label_names):
        # Get indices of images with current label
        label_indices = np.where(labels == i)[0]
        # Randomly select num_samples images with current label
        sample_indices = np.random.choice(label_indices, num_samples, replace=False)

        for j, index in enumerate(sample_indices):
            ax = axs[i, j]
            ax.imshow(images[index])
            ax.axis('off')

            # Predict label for current image
            predicted_label = predicted_labels[index]
            ax.set_title('Predicted: {}\nTrue: {}'.format(label_names[predicted_label], label_name))

    plt.tight_layout()
    plt.savefig('hog_svm_cross_val.png')
    plt.show()


def main():
    # Load dataset
    dataset_dir = './dataset_rgb'
    images, labels, label_names = load_dataset(dataset_dir)

    # Extract HOG features
    hog_features = extract_hog_features(images)

    # Initialize SVM classifier
    svm_classifier = SVC(kernel='linear')

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    predicted_labels = np.zeros_like(labels)
    for train_index, test_index in kf.split(hog_features):
        X_train, X_test = hog_features[train_index], hog_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        svm_classifier.fit(X_train, y_train)
        predicted_labels[test_index] = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predicted_labels[test_index])
        accuracies.append(accuracy)

    # Calculate and print average accuracy
    mean_accuracy = np.mean(accuracies)
    print("Accuracies:", accuracies)
    print("Average Accuracy:", mean_accuracy)

    # Visualize classification results
    visualize_classification(images, labels, label_names, predicted_labels)


if __name__ == "__main__":
    main()
