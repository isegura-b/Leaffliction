
import os
import sys

directory = sys.argv[1]

total_images = 0

for plant in os.listdir(directory):
    plant_path = os.path.join(directory, plant)

    if os.path.isdir(plant_path):
        print("\nPlant:", plant)

        for disease in os.listdir(plant_path):
            disease_path = os.path.join(plant_path, disease)

            if os.path.isdir(disease_path):
                images = os.listdir(disease_path)
                count = len(images)

                print(disease, ":", count, "images")
                total_images += count

print("\nTotal images:", total_images)