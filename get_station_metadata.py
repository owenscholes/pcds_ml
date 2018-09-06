import pandas as pd

metadata = pd.read_csv('metadata.csv')
n_ids = metadata.loc[:,'native_id']

inpt = input('station id: \n')

for i in range(len(n_ids.values)):
    if n_ids.values[i] == inpt:
        print(metadata.iloc[i])
        print(metadata.iloc[i]['lat'])
