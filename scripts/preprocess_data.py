import pandas as pd

X = pd.read_csv('/home/andrey/projects/abalone/datasets/X_abalone.csv')

X = pd.get_dummies(X, dtype=int)

X.to_csv('/home/andrey/projects/abalone/datasets/X_processed.csv',
         index=None)
