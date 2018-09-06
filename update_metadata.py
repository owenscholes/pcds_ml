import pandas as pd
from numpy import random

def has_data(record):
    try:
        size = record.size
    except AttributeError:
        return False
    if size > 20:
        return True
    else:
        return False

def get_resolution(record):
    if not has_data(record):
        return None

    resolution = []
    for x in range(10):
        res_check = 'unknown'
        index = random.randint(1,record.shape[0])
        times = record.loc[index:index+1,'obs_time']
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

def update_resolution(metadata):
    terrors = []
    ferrors = []

    for i in range(metadata.shape[0]):
        network = metadata.loc[i,'network_name']
        id = metadata.loc[i,'native_id']
        try:
            record = pd.read_pickle('D:/PCIC/record_data/'+ network + '/' + id + '.pickle')
            res = get_resolution(record)
            metadata.at[i,'resolution'] = res
        except FileNotFoundError:
            if network not in ferrors:
                ferrors.append(network)
        except TypeError:
            if network not in terrors:
                terrors.append(network)
    return metadata
    print('type errors:')
    print(terrors)
    print('file errors:')
    print(ferrors)
    print('done')

def flag_empty(metadata):
    terrors = []
    ferrors = []
    for i in range(metadata.shape[0]):
        id = metadata.at[i,'native_id']
        network = metadata.at[i,'network_name']
        try:
            record = pd.read_pickle('../station_data/'+network+'/'+id+'.pickle')
        except FileNotFoundError:
            record = None
            metadata.at[i,'flag'] = 'NO RECORD'
            if network not in ferrors:
                ferrors.append(network)
        except TypeError:
            record = None
            if network not in terrors:
                terrors.append(network)
        if has_data(record):
            metadata.at[i,'flag'] = 'good'
    #print('type errors:')
    #print(terrors)
    #print('file errors:')
    #print(ferrors)
    #print('done')
    return metadata

old_md = pd.read_csv('metadata/md_test3.csv')

new_md = flag_empty(old_md)
print('done flagging')
new_md.to_csv('metadata/md_test4.csv')
print('done writing')
