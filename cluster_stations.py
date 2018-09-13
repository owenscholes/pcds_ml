import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utm
import scipy.spatial.distance as dist
from sklearn.cluster import KMeans,Birch,DBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
import time

def to_ec_standards(record):
    fixed = ['ENV-AQN','FLNRO-WMB','EC-raw']
    if record[1]['network_name'] == 'ENV-AQN':
        temp_var = 'temp_mean'
    if record[1]['network_name'] == 'EC-raw':
        temp_var = 'temp_mean'
    if record[1]['network_name'] == 'FLNRO-WMB':
        temp_var = 'temperature'
    if record[1]['network_name'] in fixed:
        data = record[0]
        for col in data.columns:
            #print(record[0])
            #print(type(col) == type('a'))
            if 'temp' not in col:
                data = data.drop(columns=[col])
        dfmax = data.resample('D').max()
        dfmin = data.resample('D').min()
        #oldp = pd.DataFrame(data=record[0].loc[:,'precip_total'],index=record[0].loc[:,'obs_time'])
        #dfp = oldp.resample('D').sum()
        df = pd.merge(dfmax,dfmin,on='obs_time')
        #df = pd.merge(df,dfp,on='obs_time')
        try:
            df.columns = ['max_temp','min_temp']
        except ValueError:
            print(record[1])
            print(df.dtypes)
        #df.columns = ['max_temp','min_temp','one-day_precipitation']
        return (df,record[1])
    else:
        return record

def load_by_name(name):
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0
    metadata = pd.read_csv('metadata/md_current.csv')
    varnames = pd.read_csv('varnames_by_network.csv')

    #Find search term in metadata file, record station's md and native id
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

    #Get record, reindex it and format for output
    out = []
    for i, net_id in enumerate(nets_ids):
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        df = df.set_index('obs_time')
        out.append((df,md[i]))

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
            if ut[2] != target_utm[2] or ut[3] != target_utm[3]:
                continue
            coords = [ut[0],ut[1],metadata.at[row,'elev']]
            if dist.euclidean(coords,target_coords) <= rad:
                if metadata.at[row,'flag']=='good':
                    nets_ids.append((metadata.at[row,'network_name'],metadata.at[row,'native_id'],metadata.at[row,'station_name']))
                    md.append(metadata.iloc[row])
                    #print(metadata.at[row,'station_name'])
                    #print(ut)
                elif metadata.at[row,'flag']=='NO DATA':
                    ndcounter += 1
                elif metadata.at[row,'flag']=='NO RECORD':
                    nrcounter+= 1
        except (AttributeError,ValueError):
            pass

    out = []
    for i, net_id in enumerate(nets_ids):
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        df = df.set_index('obs_time')
        out.append((df,md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def plot_on_map(records,labels):
    xcoords = []
    ycoords = []
    nets = []
    for x in records:
        xcoords.append(x[1]['lon'])
        ycoords.append(x[1]['lat'])
        nets.append(x[1]['network_name'])

    plt.scatter(xcoords,ycoords,c=labels)
    plt.legend(labels)
    plt.show()

def plot_records(records,variables, ti, tf):
    '''
    t0 = (ti[0],tf[0])
    t1 = (ti[1],tf[1])
    t2 = (ti[2],tf[2])
    ti = pd.Timestamp(t0[0],t1[0],t2[0])
    tf = pd.Timestamp(t0[1],t1[1],t2[1])
    '''
    labels = []

    for variable in variables:
        for x in records:
            time_slice = x[0].loc[ti:tf]
            #if x[1]['network_name'] == 'ENV-AQN':
            #    print(time_slice.dtypes)

            if not time_slice.empty and variable in time_slice.dtypes and time_slice[variable].dtype != (object or NoneType):
                nan_check = []
                for z in range(10):
                    index = np.random.choice(time_slice.index)
                    nan_check.append(np.isnan(time_slice.at[index,variable]))
                if False in nan_check:
                    plt.scatter(time_slice.index,time_slice.loc[:,variable].values)
                    labels.append((x[1]['station_name'],x[1]['network_name'],variable))
                else:
                    print('Column was nan')
    plt.ylabel(variables)
    plt.legend(labels)
    plt.show()

def calculate_dist_matrix(records,variable):
    last = pd.Timestamp(1800,1,1)
    first = pd.Timestamp(2050,1,1)
    for x in records:
        if x[0].index[0] < first:
            first = x[0].index[0]
        if x[0].index[-1] > last:
            last = x[0].index[-1]

    length = (last - first).days
    obs_array = np.full((length+1,len(records)),None)
    for i,x in enumerate(records):
        obs_array[(x[0].index[0]-first).days:(x[0].index[0]-first).days + x[0].shape[0],i] = x[0].loc[:,variable].values
    for i in range(obs_array.shape[1]):
        if obs_array[:,i][obs_array[:,i]!=None].any():
            obs_array[:,i][obs_array[:,i]==None] = np.nanmean(obs_array[:,i][obs_array[:,i]!=None])
    print(obs_array)
    dist_mat = pairwise.pairwise_distances(obs_array)

    '''
    dist_mat = np.full((len(records),len(records)),None)
    #print(dist_mat)
    for i,rec_i in enumerate(records):
        for j,rec_j in enumerate(records):
            if i==j:
                dist_mat[i][j] = 0
            elif variable not in rec_j[0].columns:
                dist_mat[i][j] == 300
            else:
                overlap_i = rec_i[0].loc[rec_j[0].index[0]:rec_j[0].index[-1],variable]
                if overlap_i.size == 0:
                    continue
                overlap_j = rec_j[0].loc[overlap_i.index[0]:overlap_i.index[-1],variable].values
                overlap_i = overlap_i.values

                if overlap_i.size == overlap_j.size and overlap_i.size > 0:
                    #print(overlap_i,overlap_j)
                    diffs = []
                    for ind,obs in enumerate(overlap_i):
                        try:
                            if np.isnan(obs) or np.isnan(overlap_j[ind]):
                                continue
                        except TypeError:
                            pass
                        try:
                            diff = obs - overlap_j[ind]
                            diffs.append(diff)
                        except TypeError:
                            pass
                    if diffs != []:
                        diffs = np.array(diffs)
                        avg = np.mean(diffs)
                        #print(avg)
                        if not np.isnan(avg):
                            dist_mat[i][j] = abs(avg)
                elif overlap_i.size > 0:
                    #print('Problem')
                    pass
    full = dist_mat[~(dist_mat==None)]
    avg = np.mean(full)
    dist_mat[dist_mat==None] = avg
    '''
    return dist_mat

#source = load_by_coords(49.382662, -121.473297,40,100000)
#source = load_by_name('hope')+load_by_name('vancouver')+load_by_name('kamloops')+load_by_name('george')
source = load_by_name('hope')
for i,x in enumerate(source):
    source[i] = to_ec_standards(x)
    #print(x[1]['network_name'])
safe_source = []
for x in source:
    if x[1]['network_name'] in ('EC','BCH','ENV-AQN','FLNRO-WMB'):
        safe_source.append(x)
dist_matrix = calculate_dist_matrix(safe_source,'max_temp')
'''
for element in dist_matrix.flatten():
    if element != None:
        if np.isnan(element):
            print('!!!!!!!!!')
'''

#print(dist_matrix)

agg = AgglomerativeClustering(n_clusters=10,affinity='precomputed',linkage='complete')
labels = agg.fit_predict(dist_matrix)
plot_on_map(safe_source,labels)
print(labels)

variables = ['max_temp']
ti = pd.Timestamp(1970,8,1)
tf = pd.Timestamp(2018,8,1)
#plot_records(source,variables,ti,tf)
'''
for val in range(10):
    sources = []
    for ind,x in enumerate(labels):
        if x == val:
            sources.append(safe_source[ind])
    plot_records(sources,variables,ti,tf)
'''
'''
dbs = DBSCAN(min_samples=3,eps=0.2,metric='precomputed')
clusters = dbs.fit(dist_matrix)
plot_on_map(safe_source,clusters.labels_)
print(clusters.labels_)
'''




'''

'''
