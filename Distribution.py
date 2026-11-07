
import os
import sys
import matplotlib.pyplot as plt

directory = sys.argv[1]

apple = {}
grape = {}
total_images = 0

for disease in os.listdir(directory):
    path = os.path.join(directory, disease)

    if os.path.isdir(path):
        count = len(os.listdir(path))

        print(disease, ":", count, "images")
        total_images += count

        if disease.startswith("Apple"):
            apple[disease] = count
        elif disease.startswith("Grape"):
            grape[disease] = count

print("\nTotal images:", total_images)

# Apple
plt.subplot(1, 2, 1)
plt.bar(apple.keys(), apple.values())
plt.title("Apple - Images per disease")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.pie(apple.values(), labels=apple.keys(), autopct="%1.1f%%")
plt.title("Apple - Distribution")

plt.tight_layout()
plt.show()

# Grape
plt.subplot(1, 2, 1)
plt.bar(grape.keys(), grape.values())
plt.title("Grape - Images per disease")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.pie(grape.values(), labels=grape.keys(), autopct="%1.1f%%")
plt.title("Grape - Distribution")

plt.tight_layout()
plt.show()