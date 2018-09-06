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

def load_by_name(name):
    records = []
    nets_ids_names = []
    ndcounter = 0
    nrcounter = 0
    metadata = pd.read_csv('metadata/md_current.csv')

    for row in range(metadata.shape[0]):
        try:
            if name in metadata.at[row,'station_name'].lower():
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
    return records

def get_md_by_name(name):
    metadata = pd.read_csv('metadata/md_current.csv')
    names = metadata.loc[:,'station_name']

    for i in range(len(names.values)):
        if names.values[i] == name:
            return metadata.iloc[i]

def plot_records(records,variable, ti, tf):
    t0 = (ti[0],tf[0])
    t1 = (ti[1],tf[1])
    t2 = (ti[2],tf[2])
    ti = pd.Timestamp(t0[0],t1[0],t2[0])
    tf = pd.Timestamp(t0[1],t1[1],t2[1])

    for x in records:
        time_slice = x.loc[x['obs_time'] < tf]
        time_slice = time_slice.loc[time_slice['obs_time'] > ti]

        if not time_slice.empty and variable in time_slice.dtypes:
            if time_slice[variable].dtype != object:
                plt.scatter(time_slice.loc[:,'obs_time'].values,time_slice.loc[:,variable].values)
    plt.show()

plot_records(load_by_name('princeton'),'one_day_rain',(2005,4,6),(2007,3,7))
