import os
import numpy as np
from skimage import io, color, feature, transform
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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


def train_svm_classifier(X_train, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def visualize_classification(images, labels, label_names, classifier):
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
            hog_feature = extract_hog_features([images[index]])
            predicted_label = classifier.predict(hog_feature)
            ax.set_title('Predicted: {}\nTrue: {}'.format(label_names[predicted_label[0]], label_name))

    plt.tight_layout()
    plt.show()


def main():
    # Load dataset
    dataset_dir = './dataset_rgb'
    images, labels, label_names = load_dataset(dataset_dir)

    # Extract HOG features
    hog_features = extract_hog_features(images)

    # Initialize and train SVM classifier
    svm_classifier = train_svm_classifier(hog_features, labels)

    # Visualize classification
    visualize_classification(images, labels, label_names, svm_classifier)

    # Evaluate classifier
    evaluate_classifier(svm_classifier, hog_features, labels)


if __name__ == "__main__":
    main()
