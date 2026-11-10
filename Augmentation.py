
import os
import sys
import cv2
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

OUTPUT = "augmented_directory"
EFFECTS = ["Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion"]


def augment(image, effect):
    height, width = image.shape[:2]

    if effect == "Flip":
        return cv2.flip(image, 1)

    if effect == "Rotate":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if effect == "Skew":
        points1 = np.float32([
            [0, 0], [width, 0],
            [0, height], [width, height]
        ])
        points2 = np.float32([
            [width * 0.2, 0], [width, 0],
            [0, height], [width * 0.8, height]
        ])
        matrix = cv2.getPerspectiveTransform(points1, points2)
        return cv2.warpPerspective(image, matrix, (width, height))

    if effect == "Shear":
        matrix = np.float32([[1, 0.2, 0], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (width, height))

    if effect == "Crop":
        crop = image[
            height // 10:height * 9 // 10,
            width // 10:width * 9 // 10
        ]
        return cv2.resize(crop, (width, height))

    if effect == "Distortion":
        map_x, map_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )
        map_x = map_x + 5 * np.sin(map_y / 10)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def display(image):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for axis, effect in zip(axes.flat, EFFECTS):
        result = augment(image, effect)
        axis.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axis.set_title(effect)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def balance_dataset(directory):
    classes = {}

    # From Distribution: directory traversal and image counting
    for disease in os.listdir(directory):
        path = os.path.join(directory, disease)

        if os.path.isdir(path):
            images = [
                os.path.join(path, file)
                for file in os.listdir(path)
                if file.lower().endswith(".jpg")
            ]

            if images:
                classes[disease] = images

    if not classes:
        print("No images found")
        return

    target = max(len(images) for images in classes.values())

    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)

    print("\nTarget:", target, "images per class\n")

    for disease, images in classes.items():
        destination = os.path.join(OUTPUT, disease)
        os.makedirs(destination, exist_ok=True)

        for image_path in images:
            shutil.copy2(image_path, destination)

        missing = target - len(images)

        for i in range(missing):
            image_path = random.choice(images)
            image = cv2.imread(image_path)

            effect = EFFECTS[i % len(EFFECTS)]
            result = augment(image, effect)

            name = os.path.splitext(os.path.basename(image_path))[0]
            filename = name + "_" + effect + "_" + str(i + 1) + ".jpg"

            cv2.imwrite(os.path.join(destination, filename), result)

        print(
            disease, ":",
            len(images), "originals +",
            missing, "new =",
            target
        )

    print("\nBalanced dataset saved in", OUTPUT)

    first_image = cv2.imread(next(iter(classes.values()))[0])
    display(first_image)


if len(sys.argv) != 2:
    print("Usage: python Augmentation.py directory")
    sys.exit(1)

if os.path.isdir(sys.argv[1]):
    balance_dataset(sys.argv[1])
else:
    print("Invalid directory")
