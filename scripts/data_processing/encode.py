import pandas as pd
import os

def text_to_numeric(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(input_file)

    categorical_cols = ['country', 'store', 'product']

    df = pd.get_dummies(df, columns=categorical_cols)

    output_file = os.path.join(output_folder, 'train.csv')
    df.to_csv(output_file, index=False)
    print(f"File saved to {output_file}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_to_numeric.py <input_csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = 'data/prepared'
    text_to_numeric(input_csv, output_dir)
