import csv
import os

def get_features(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, mode='r', encoding='utf-8') as file, \
         open(os.path.join(output_folder, 'train.csv'), mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(file)
        fieldnames = ['country', 'store', 'product', 'num_sold']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:
            writer.writerow({field: row[field] for field in fieldnames})

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python getFeatures.py <input_csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = 'data/features'
    get_features(input_csv, output_dir)
