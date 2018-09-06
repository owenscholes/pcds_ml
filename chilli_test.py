import pandas as pd
import matplotlib.pyplot as plt

def has_data(station):
    if station.size > 20 and not station.empty:
        return True
    else:
        return False

stations = []

for x in [30,40,45,61,64,65,67,77,80]:
    id = '11015' + str(x)
    df = pd.read_pickle('../station_data/ec/'+id+'.pickle')
    if has_data(df):
        stations.append(df)

for a in stations:
    b = a.loc[a['obs_time'] > pd.Timestamp(1990,1,1)]
    c = b.loc[b['obs_time'] < pd.Timestamp(1991,1,1)]
    plt.scatter(c.loc[:,'obs_time'].values, c.loc[:,'max_temp'].values)

plt.show()
