import json
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import os

def evaluate_model(model_file, test_data_file, output_file):

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = joblib.load(model_file)

    test_df = pd.read_csv(test_data_file)
    X_test = test_df.drop('num_sold', axis=1)
    y_test = test_df['num_sold']

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    with open(output_file, 'w') as file:
        json.dump({'mse': mse}, file)
    
    print(f"Model evaluation saved to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model-file test-data-file output-file\n")
        sys.exit(1)

    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_json_path = sys.argv[3]

    evaluate_model(model_path, test_data_path, output_json_path)
