import cv2
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt


def segment(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int16)
    warmth = np.clip(lab[:, :, 2] - lab[:, :, 1] + 128, 0, 255)
    blur = cv2.GaussianBlur(warmth.astype(np.uint8), (5, 5), 0)
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(mask), None

    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask, contour


def find_defects(image, mask):
    green = cv2.GaussianBlur(
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 1], (5, 5), 0
    )
    limit, _ = cv2.threshold(
        green[mask > 0].reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    spots = ((green > limit) & (mask > 0)).astype(np.uint8) * 255
    spots = cv2.morphologyEx(
        spots, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    contours, _ = cv2.findContours(
        spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contours if cv2.contourArea(c) >= 15]


def transform(image):
    mask, contour = segment(image)

    spots = find_defects(image, mask) if contour is not None else []
    healthy = mask.copy()
    cv2.drawContours(healthy, spots, -1, 0, cv2.FILLED)

    masked = np.full_like(image, 255)
    masked[healthy > 0] = image[healthy > 0]

    roi = image.copy()
    analyze = image.copy()
    landmarks = image.copy()

    if contour is not None:
        roi[healthy > 0] = (0, 255, 0)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(roi, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 4)

        cv2.drawContours(analyze, spots, -1, (255, 0, 0), 2)
        cv2.drawContours(analyze, [contour], -1, (255, 0, 255), 3)
        cx, cy = x + w // 2, y + h // 2
        cv2.line(analyze, (cx, y), (cx, y + h), (255, 0, 255), 2)
        cv2.line(analyze, (x, cy), (x + w, cy), (255, 0, 255), 2)
        cv2.putText(
            analyze, "Area: " + str(int(cv2.contourArea(contour))),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        for spot in spots:
            moments = cv2.moments(spot)
            center = (int(moments["m10"] / moments["m00"]),
                      int(moments["m01"] / moments["m00"]))
            cv2.circle(landmarks, center, 4, (0, 128, 255), -1)

    results = {
        "Gaussian_blur": cv2.GaussianBlur(healthy, (15, 15), 0),
        "Mask": masked,
        "ROI_objects": roi,
        "Analyze_object": analyze,
        "Pseudolandmarks": landmarks
    }
    return results, mask


CHANNELS = [
    ("blue", cv2.COLOR_BGR2RGB, 2, "blue"),
    ("blue-yellow", cv2.COLOR_BGR2LAB, 2, "gold"),
    ("green", cv2.COLOR_BGR2RGB, 1, "green"),
    ("green-magenta", cv2.COLOR_BGR2LAB, 1, "magenta"),
    ("hue", cv2.COLOR_BGR2HSV, 0, "blueviolet"),
    ("lightness", cv2.COLOR_BGR2LAB, 0, "dimgray"),
    ("red", cv2.COLOR_BGR2RGB, 0, "red"),
    ("saturation", cv2.COLOR_BGR2HSV, 1, "cyan"),
    ("value", cv2.COLOR_BGR2HSV, 2, "orange"),
]


def histogram(image, mask, axis):
    pixels = max(1, int(np.count_nonzero(mask)))

    for name, conversion, index, color in CHANNELS:
        channel = cv2.cvtColor(image, conversion)[:, :, index]
        hist = cv2.calcHist([channel], [0], mask, [256], [0, 256])
        axis.plot(hist * 100 / pixels, color=color, label=name)

    axis.set_title("Color histogram")
    axis.set_xlabel("Pixel intensity")
    axis.set_ylabel("Proportion of pixels (%)")
    axis.legend(title="color Channel", fontsize=7)


def show_results(image, mask, results):
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

    histogram(image, mask, axes.flat[6])
    axes.flat[7].axis("off")

    plt.tight_layout()
    plt.show()


def save_results(image, mask, results, output_dir, name, mask_only=False):
    os.makedirs(output_dir, exist_ok=True)

    for effect, result in results.items():
        if mask_only and effect != "Mask":
            continue

        filename = name + "_" + effect + ".jpg"
        cv2.imwrite(os.path.join(output_dir, filename), result)

    if not mask_only:
        fig, axis = plt.subplots()
        histogram(image, mask, axis)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, name + "_Color_histogram.png"))
        plt.close()


parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?")
parser.add_argument("-src")
parser.add_argument("-dst", default="transformations")
parser.add_argument("-mask", action="store_true",
                    help="Save only image masks")

args = parser.parse_args()


if args.image:
    image = cv2.imread(args.image)

    if image is not None:
        results, mask = transform(image)
        show_results(image, mask, results)
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
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(current, filename)
                image = cv2.imread(path)

                if image is not None:
                    results, mask = transform(image)
                    name = os.path.splitext(filename)[0]

                    save_results(
                        image, mask, results, output_dir, name, args.mask
                    )
                    count += 1

    print(count, "images processed")
    print("Results saved in:", destination)

else:
    parser.print_help()
