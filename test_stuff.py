import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

import utm
import scipy.spatial.distance as dist

def has_data(record):
    try:
        size = record.size
    except AttributeError:
        return False
    if size > 20:
        return True
    else:
        return False

def load_by_nid(network,nid):
    df = pd.read_pickle('D:/PCIC/station_data/' + network + '/' + nid + '.pickle')
    metadata = pd.read_csv('metadata/md_current.csv')
    for row in range(metadata.shape[0]):
        try:
            if metadata.at[row,'native_id'] == nid:
                if metadata.at[row,'flag']=='good':
                    station_md = metadata.loc[row,:]
                elif metadata.at[row,'flag']=='NO DATA':
                    print('Record is blank')
        except AttributeError:
            pass
    #station_md = metadata.loc[metadata['native_id']==nid]
    record = [[df,station_md]]

    return record

def load_by_name(name):
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0
    metadata = pd.read_csv('metadata/md_current.csv')
    varnames = pd.read_csv('varnames_by_network.csv')

    for row in range(metadata.shape[0]):
        try:
            if name in metadata.at[row,'station_name'].lower():
                if metadata.at[row,'flag']=='good':
                    nets_ids.append((metadata.at[row,'network_name'],metadata.at[row,'native_id'],metadata.at[row,'station_name']))
                    md.append(metadata.iloc[row])
                elif metadata.at[row,'flag']=='NO DATA':
                    ndcounter += 1
                elif metadata.at[row,'flag']=='NO RECORD':
                    nrcounter+= 1
        except AttributeError:
            pass

    for net_id in nets_ids:
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        if 'tm' in df.columns:
            df.rename(columns={'tm':'temp_mean'}, inplace=True)
        if 'current_air_temperature1' in df.columns:
            df.rename(columns={'current_air_temperature1':'temp_mean'}, inplace=True)
        if 'current_air_temperature2' in df.columns:
            df.rename(columns={'current_air_temperature2':'temp_mean'}, inplace=True)
        if 'air_temperature' in df.columns:
            df.rename(columns={'air_temperature':'temp_mean'}, inplace=True)
        if 'airtemp' in df.columns:
            df.rename(columns={'airtemp':'temp_mean'}, inplace=True)
        if 'tx' in df.columns:
            df.rename(columns={'tx':'max_temp'}, inplace=True)
        if 'maximum_air_temperature' in df.columns:
            df.rename(columns={'maximum_air_temperature':'max_temp'}, inplace=True)
        if 'tmax' in df.columns:
            df.rename(columns={'tmax':'max_temp'}, inplace=True)
        records.append(df)
    out = []
    for i in range(len(md)):
        out.append((records[i],md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def load_by_coords(lat,long,elev,rad):
    target_utm = utm.from_latlon(lat,long)
    target_coords = [target_utm[0],target_utm[1],elev]
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0
    metadata = pd.read_csv('metadata/md_current.csv')

    for row in range(metadata.shape[0]):
        try:
            ut = utm.from_latlon(metadata.at[row,'lat'],metadata.at[row,'lon'])
            coords = [ut[0],ut[1],metadata.at[row,'elev']]
            if dist.euclidean(coords,target_coords) <= rad:
                if metadata.at[row,'flag']=='good':
                    nets_ids.append((metadata.at[row,'network_name'],metadata.at[row,'native_id'],metadata.at[row,'station_name']))
                    md.append(metadata.iloc[row])
                elif metadata.at[row,'flag']=='NO DATA':
                    ndcounter += 1
                elif metadata.at[row,'flag']=='NO RECORD':
                    nrcounter+= 1
        except (AttributeError,ValueError):
            pass

    for net_id in nets_ids:
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        records.append(df)
    out = []
    for i in range(len(md)):
        out.append((records[i],md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def how_much_nan(records):
    #Doesn't work
    for x in records:
        count = []
        if x[1]['network_name'] == 'EC':
            print('EC')
            x[0].at[3,'obs_time'] = math.nan
        for y in x[0].columns:
            try:
                if math.nan in np.unique(x[0].loc[:,y].values) and np.unique(x[0].loc[:,y].values).size > 1:
                    count.append(True)
            except TypeError:
                pass
        if True in count:
            x[0]['has_both'] = True
        else:
            x[0]['has_both'] = False
        print(x[0].at[0,'has_both'])

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
    labels = []

    for x in records:
        time_slice = x[0].loc[x[0]['obs_time'] < tf]
        time_slice = time_slice.loc[time_slice['obs_time'] > ti]



        if not time_slice.empty and variable in time_slice.dtypes and time_slice[variable].dtype != (object or NoneType):
            nan_check = []
            for z in range(10):
                index = np.random.choice(time_slice.index)
                nan_check.append(math.isnan(time_slice.at[index,variable]))
            if False in nan_check:
                plt.scatter(time_slice.loc[:,'obs_time'].values,time_slice.loc[:,variable].values)
                labels.append((x[1]['station_name'],x[1]['network_name']))
    plt.ylabel(variable)
    plt.legend(labels)
    #plt.show()





#source = load_by_coords(49.698474, -123.158422,5,30000)
source = load_by_name('squamish')
#source = load_by_nid('ec_raw','1048899')
variable = 'max_temp'
plot_records(source,variable,(2005,4,30),(2006,4,30))
plt.show()
