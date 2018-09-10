import pandas as pd
from os import listdir
from numpy import unique

def discover_varnames():
    vars_by_net = {}
    for network in reversed(listdir('../station_data')):
        vars_by_net[network] = []
        for station in listdir('../station_data/' + network):
            df = pd.read_pickle('../station_data/' + network + '/' + station)
            for var in df.columns:
                if var not in vars_by_net[network]:
                    vars_by_net[network].append(var)
        print('finished {}'.format(network))

    longest = 0
    for net in listdir('../station_data'):
        if len(vars_by_net[net]) > longest:
            longest = len(vars_by_net[net])
    print(longest)

    for net in listdir('../station_data'):
        while len(vars_by_net[net]) < longest:
            vars_by_net[net].append(None)

    output = pd.DataFrame(data=vars_by_net)

    output.to_csv('varnames_by_network.csv')
    print('done writing')

#Doesn't work yet, might not need it
def get_res_by_network():
    stuff = {}
    for net in varnames.columns:
        if net == 'ARDA':
            continue
        selection = metadata.loc[(metadata['network_name']==net) & (metadata['flag']=='good')]
        res = unique(selection['resolution'].values)
        stuff[net] = res

    longest = 0
    for x in varnames.columns:
        if net == 'ARDA':
            continue
        if len(stuff[x]) > longest:
            longest = stuff[x]
    print(longest)

    for x in varnames.columns:
        if net == 'ARDA':
            continue
        while len(stuff[x]) < longest:
            stuff[x].append(None)


    df = pd.DataFrame(data=stuff)
    return df

metadata = pd.read_csv('metadata/md_current.csv')
varnames = pd.read_csv('varnames_by_network.csv')

#df = get_res_by_network()
#print(df)
