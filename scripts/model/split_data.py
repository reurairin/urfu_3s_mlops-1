import yaml
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["split"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython split_data.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output_train = os.path.join("data", "split", "train.csv")
os.makedirs(os.path.join("data", "split"), exist_ok=True)
f_output_test = os.path.join("data", "split", "test.csv")
os.makedirs(os.path.join("data", "split"), exist_ok=True)

p_split_ratio = params["split_ratio"]

df = pd.read_csv(f_input)

X = df.drop('num_sold', axis=1)
y = df['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_split_ratio, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(f_output_train, index=False)
test_df.to_csv(f_output_test, index=False)
