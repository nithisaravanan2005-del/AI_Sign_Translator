import os
import csv
import numpy as np

RAW_DATA_PATH = "data/raw"
OUTPUT_FILE = "data/dataset.csv"

data = []
labels = []

# Loop through each sign folder
for sign_name in os.listdir(RAW_DATA_PATH):
    sign_path = os.path.join(RAW_DATA_PATH, sign_name)

    if os.path.isdir(sign_path):
        for file in os.listdir(sign_path):
            file_path = os.path.join(sign_path, file)

            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                row = next(reader)

                row = list(map(float, row))
                data.append(row)
                labels.append(sign_name)

# Save combined dataset
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)

    for i in range(len(data)):
        writer.writerow(data[i] + [labels[i]])

print("Dataset created successfully.")
print("Total samples:", len(data))