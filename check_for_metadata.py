import pandas as pd
from os import listdir

metadata = pd.read_csv('metadata.csv')
n_ids = metadata.loc[:,'native_id']

for network in listdir('../station_data'):
    counter = 0
    no_md = 0

    for station in listdir('../station_data/' + network):
        id = station.rstrip('.pickle')
        has_md = False
        counter += 1
        if id in n_ids.values:
            has_md = True
        if has_md == False:
            no_md += 1
    have = counter - no_md
    print(network + '\t{}/{}'.format(have,counter))
