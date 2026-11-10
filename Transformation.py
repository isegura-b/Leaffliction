
import cv2
import os
import argparse
import matplotlib.pyplot as plt


def transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(image, (15, 15), 0)

    # Mask
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find objects
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    roi = image.copy()
    analyze = image.copy()
    landmarks = image.copy()

    if contours:
        contour = max(contours, key=cv2.contourArea)

        # ROI objects
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]

        # Analyze object
        area = cv2.contourArea(contour)
        cv2.drawContours(analyze, [contour], -1, (0, 255, 0), 2)

        cv2.putText(
            analyze, "Area: " + str(int(area)),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )

        # Pseudolandmarks
        step = max(1, len(contour) // 10)

        for point in contour[::step]:
            x, y = point[0]
            cv2.circle(landmarks, (int(x), int(y)), 5, (0, 0, 255), -1)

    return {
        "Gaussian_blur": blur,
        "Mask": mask,
        "ROI_objects": roi,
        "Analyze_object": analyze,
        "Pseudolandmarks": landmarks
    }


def histogram(image, axis):
    for channel, color in enumerate(["blue", "green", "red"]):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        axis.plot(hist, color=color)

    axis.set_title("Color histogram")


def show_results(image, results):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes.flat[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes.flat[0].set_title("Original")
    axes.flat[0].axis("off")

    for axis, (name, result) in zip(axes.flat[1:6], results.items()):
        if len(result.shape) == 2:
            axis.imshow(result, cmap="gray")
        else:
            axis.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        axis.set_title(name)
        axis.axis("off")

    histogram(image, axes.flat[6])
    axes.flat[7].axis("off")

    plt.tight_layout()
    plt.show()


def save_results(image, results, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)

    for effect, result in results.items():
        filename = name + "_" + effect + ".jpg"
        cv2.imwrite(os.path.join(output_dir, filename), result)

    fig, axis = plt.subplots()
    histogram(image, axis)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name + "_Color_histogram.png"))
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?")
parser.add_argument("-src")
parser.add_argument("-dst", default="transformations")

args = parser.parse_args()


if args.image:
    image = cv2.imread(args.image)

    if image is not None:
        results = transform(image)
        show_results(image, results)
    else:
        print("Image not found")

elif args.src:
    if not os.path.isdir(args.src):
        print("Source directory not found")
        sys.exit(1)

    source = os.path.abspath(args.src)
    destination = os.path.abspath(args.dst)

    if os.path.commonpath([source, destination]) == source:
        print("Destination must be outside the source directory")
        sys.exit(1)

    count = 0

    for current, dirs, files in os.walk(source):
        relative = os.path.relpath(current, source)
        output_dir = os.path.join(destination, relative)

        for filename in files:
            if filename.lower().endswith(".jpg"):
                path = os.path.join(current, filename)
                image = cv2.imread(path)

                if image is not None:
                    results = transform(image)
                    name = os.path.splitext(filename)[0]

                    save_results(image, results, output_dir, name)
                    count += 1

    print(count, "images processed")
    print("Results saved in:", destination)

else:
    parser.print_help()
