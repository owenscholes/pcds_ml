import pandas as pd
import matplotlib.pyplot as plt

def has_data(record):
    try:
        size = record.size
    except AttributeError:
        return False
    if size > 20:
        return True
    else:
        return False

records = []
nets_ids_names = []
ndcounter = 0
nrcounter = 0
metadata = pd.read_csv('metadata/md_current.csv')

while True:
    inpt = input('Search records by name or network?\n')
    if inpt in ('name','network'):
        break

search = input('Search term: ')

if inpt == 'network':
    selection = metadata.loc[metadata['network_name']==search]
    for row in range(selection.shape[0]):
        if metadata.at[row,'flag']=='good':
            nets_ids_names.append((search,selection.at[row,'native_id'],selection.at[row,'station_name']))
        elif metadata.at[row,'flag']=='NO DATA':
            ndcounter += 1
        elif metadata.at[row,'flag']=='NO RECORD':
            nrcounter+= 1

if inpt == 'name':
    for row in range(metadata.shape[0]):
        try:
            if search in metadata.at[row,'station_name'].lower():
                if metadata.at[row,'flag']=='good':
                    nets_ids_names.append((metadata.at[row,'network_name'],metadata.at[row,'native_id'],metadata.at[row,'station_name']))
                elif metadata.at[row,'flag']=='NO DATA':
                    ndcounter += 1
                elif metadata.at[row,'flag']=='NO RECORD':
                    nrcounter+= 1
        except AttributeError:
            pass

for net_id in nets_ids_names:
    df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
    df.at[0,'station_name'] = net_id[2]
    df.at[0,'network_name'] = net_id[0]
    records.append(df)

length = len(records)
print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
