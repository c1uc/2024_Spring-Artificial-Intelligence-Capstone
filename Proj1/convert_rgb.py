import os
from PIL import Image


def convert_to_rgb(input_folder, output_folder):
    count = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for label_name in os.listdir(input_folder):
        label_input_folder = os.path.join(input_folder, label_name)
        label_output_folder = os.path.join(output_folder, label_name)
        if not os.path.exists(label_output_folder):
            os.makedirs(label_output_folder)

        for filename in os.listdir(label_input_folder):
            input_path = os.path.join(label_input_folder, filename)
            output_path = os.path.join(label_output_folder, filename)

            # Open the image
            img = Image.open(input_path)

            # Convert RGBA to RGB (ignoring alpha channel)
            img = img.convert("RGB")

            # Resize the image
            img = img.resize((180, 180))

            # Save the image
            img.save(output_path)
            count += 1
    return count


# Input and output directories
input_folder = './dataset_with_alpha'
output_folder = './dataset_rgb'

# Convert images
count = convert_to_rgb(input_folder, output_folder)
print(f"Converted {count} images from RGBA to RGB format.")
