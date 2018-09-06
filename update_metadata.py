import pandas as pd
from numpy import random

def has_data(station):
    if station.size > 20:
        return True
    else:
        return False

def get_resolution(station):
    if not has_data(station):
        return None

    resolution = []
    for x in range(10):
        res_check = 'unknown'
        index = random.randint(1,station.shape[0])
        times = station.loc[index:index+1,'obs_time']
        try:
            if times.iloc[0].hour != times.iloc[1].hour:
                res_check = 'hourly'
            elif times.iloc[0].day != times.iloc[1].day:
                res_check = 'daily'
            elif times.iloc[0].month != times.iloc[1].month:
                res_check = 'monthly'
        except IndexError:
            pass
        resolution.append(res_check)

    if 'unknown' not in resolution:
        res = resolution[0]
    else:
        res = 'unknown'
    return res

metadata = pd.read_csv('metadata.csv')
metadata['resolution'] = None
terrors = []
ferrors = []

for i in range(metadata.shape[0]):
    network = metadata.loc[i,'network_name']
    id = metadata.loc[i,'native_id']
    try:
        station = pd.read_pickle('D:/PCIC/station_data/'+ network + '/' + id + '.pickle')
        res = get_resolution(station)
        metadata.at[i,'resolution'] = res
    except FileNotFoundError:
        if network not in ferrors:
            ferrors.append(network)
    except TypeError:
        if network not in terrors:
            terrors.append(network)


metadata.to_csv('md_test2.csv')

print('type errors:')
print(terrors)
print('file errors:')
print(ferrors)
print('done')
