import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans,Birch,DBSCAN
from sklearn.preprocessing import StandardScaler

def load_by_nid(network,nid):
    df = pd.read_pickle('D:/PCIC/station_data/' + network + '/' + nid + '.pickle')
    df = df.set_index('obs_time')
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

def plot_records(records,variables, ti, tf,color='b'):
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
            #print(type(x))
            time_slice = x[0].loc[ti:tf]



            if not time_slice.empty and variable in time_slice.dtypes and time_slice[variable].dtype != (object or NoneType):
                nan_check = []
                for z in range(10):
                    index = np.random.choice(time_slice.index)
                    nan_check.append(np.isnan(time_slice.at[index,variable]))
                if False in nan_check:
                    plt.scatter(time_slice.index,time_slice.loc[:,variable].values,c=color)
                    labels.append((x[1]['station_name'],x[1]['network_name'],variable))
                else:
                    print('Column was nan')
    plt.ylabel(variables)
    plt.legend(labels)
    plt.show()

def cluster(records,variable,ti,tf):
    time_slice = records[0][0].loc[ti:tf]
    data = time_slice.loc[:,variable].values
    data[data==None] = np.nan
    data = data.astype('float64')
    ind = time_slice.index.values[~np.isnan(data)]
    data = data[~np.isnan(data)]
    data = np.array(list(zip(ind,data)))

    data = StandardScaler().fit_transform(data)

    clusters = DBSCAN(eps=0.5).fit(data) #for fahrenheit issue
    #clusters = DBSCAN(eps=0.7).fit(data) #for flatline

    return clusters


def remove_outliers(record,clusters,variable,ti,tf):
    time_slice = record[0][0].loc[ti:tf]
    time_slice = time_slice.loc[~np.isnan(time_slice.loc[:,variable].values)]
    for ix, obs_time in enumerate(time_slice.index):
        if clusters.labels_[ix] == -1:
            time_slice.at[obs_time,variable] = None
    return [(time_slice,record[0][1])]


#source = load_by_coords(49.788064, -126.051257,5,30000)
#source = load_by_coords(49.42805,-123.56484,543,30000)
#source = load_by_name('elphin')  #flatline
#source = load_by_nid('env-aqn', 'E208096')  #fahrenheit
#source = load_by_nid('ec','110FAG9')
source = [load_by_name('squamish')[7]]
variable = 'max_temp'
#variables = ['temperature','temp_mean']
ti = pd.Timestamp(1980,8,1)
tf = pd.Timestamp(2010,1,1)
#plot_records(source,variables,ti,tf)
#for i,x in enumerate(source):
#    source[i] = to_ec_standards(x)
clusters = cluster(source,variable,ti,tf)
#clusters = cluster(load_by_name('elphin')[0],'temperature',pd.Timestamp(2009,10,1),pd.Timestamp(2009,11,10))

cleaned = remove_outliers(source,clusters,variable,ti,tf)
plot_records(source,[variable],ti,tf)
plot_records(cleaned,[variable],ti,tf,color=clusters.labels_)
