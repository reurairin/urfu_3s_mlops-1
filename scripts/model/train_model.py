import yaml
import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model(input_file, output_folder, params):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(input_file)
    X_train = df.drop('num_sold', axis=1)
    y_train = df['num_sold']

    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )
    
    model.fit(X_train, y_train)

    model_file = os.path.join(output_folder, 'model.pkl')
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train_model.py data-file\n")
        sys.exit(1)

    params = yaml.safe_load(open("params.yaml"))["train"]
    input_csv = sys.argv[1]
    output_dir = 'models'
    train_model(input_csv, output_dir, params)
