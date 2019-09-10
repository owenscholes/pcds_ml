import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
import time
from os import listdir
import csv
import pickle

class Record:
    def __init__(self,data,md):
        self.data = data
        self.md = md
        self.ti = self.data.index[0]
        self.tf = self.data.index[-1]
        self.stats = None
        self.monthly_means = {'max_temp':{},'min_temp':{},'one_day_precipitation':{}}
        self.monthly_sds = {'max_temp':{},'min_temp':{},'one_day_precipitation':{}}
        self.neighbors = []

    def plot_output(self):
        pred_slice = self.data.loc[self.data['flag']!='M']
        out_slice = self.data.loc[self.data['flag']=='O']

        plt.scatter(self.data.index,self.data['observation'],c='k')
        plt.scatter(pred_slice.index,pred_slice['final'],c='g')
        plt.scatter(out_slice.index,out_slice['observation'],c='r')
        plt.legend(['Measured','Predicted','Outliers'])
        plt.title('{}, {}, {}'.format(self.md['station_name'],self.md['network_name'],self.stats['variable'].values))
        plt.show()

def load_correlated_gapless(network,nid,variable,corr_threshhold):
    corrs = correlations.loc[:,(network,nid,variable)]
    nearby = []
    for net,id in corrs.index:
        score = corrs[(net,id)]
        if score > corr_threshhold:
            nearby.append((net,id,score))
    records = []
    for n_net,n_nid,score in nearby:
        df = pd.read_pickle('D:/PCIC/station_data_gapless/' + n_net + '/' + n_nid + '.pickle')
        records.append((df,score,n_net,n_nid))
    return records,nearby

def name_lookup(id):
    station_name = None
    for row in range(metadata.shape[0]):
        try:
            if metadata.at[row,'native_id'] == id[1] and metadata.at[row,'network_name'] == id[0]:
                station_name = metadata.at[row,'station_name']
                break
        except AttributeError:
            continue
    return station_name

def add_nan(record,slice_index):
    if slice_index.shape[0] != record.shape[0]:
        for z in slice_index:
            if z not in record.index:
                record.at[z,:] = float('NaN')
        record.sort_index(axis=0,inplace=True)

    return record

def drop_missing(data):
    ycounter = 0
    ncounter = 0
    for row in data.index:
        if True in pd.isnull(data.loc[row,'observation':]).values or data.at[row,'flag'] in ['R','O']:
            ncounter += 1
            data.drop(labels=row,inplace=True)
        else:
            ycounter += 1
    return (data,ncounter/(ncounter+ycounter))

def simple_plot(record,variable):
    plt.scatter(record[0].index,record[0][variable],c='k')
    plt.title(record[1]['station_name'])
    plt.show()

def take_second(elem):
    return(elem[1])

def get_jd(date):
    jd = ytd_length[date.month] + date.day
    return jd

def add_window(data,testing):
    outliers = 0
    #Add empty columns to hold window of previous days
    for num in range(window_size):
        data['p{}'.format(num)] = pd.Series(np.nan,index=data.index)
    #Propogate values diagonally downward to fill empty columns
    for row in data.index:
        cell = data.at[row,'observation']
        if np.isnan(cell):
            data.at[row,'flag'] = 'X'
        else:
            try:
                mean = target.monthly_means[variable][row.month]
                std = target.monthly_sds[variable][row.month]
                if (cell <= mean - qc_level * std or cell >= mean + qc_level * std):
                    data.at[row,'flag'] = 'O'
                    outliers += 1
                    for offset in range(window_size):
                        data.at[row+(1+offset)*d,'p{}'.format(offset)] = np.nan
            except KeyError:
                pass
        #Remove random subset of points for testing
        if testing:
            if not np.isnan(cell) and row != ti:
                if np.random.random() <= missing_fraction:
                    removed_dates.append(row)
                    removed_vals.append(cell)
                    cell = np.nan
                    data.at[row,'flag'] = 'R'
        for offset in range(window_size):
            data.at[row+(1+offset)*d,'p{}'.format(offset)] = cell

    data.drop(index=data.index[data.index.shape[0]-window_size:],inplace=True) #Drop extra rows created at the end

    return data,outliers

def generate_test_stations():
    test_stations = []
    for network in listdir('../station_data_gapless'):
        stations = listdir('../station_data_gapless' + '/' + network)
        for x in range(len(stations) // 10):
            nid = stations.pop(np.random.randint(0,high=len(stations)))
            test_stations.append((network,nid.rstrip('.pickle')))
    with open('test_stations.csv', 'w',newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(test_stations)
    return(test_stations)

output_statistics = ['station_name','variable','result','neighbors',
                    'record_length','total_time','missing_fraction','measured_var',
                    'predicted_var','imputed_var','measured_mean','predicted_mean',
                    'neighbor_fill_average','prediction_fail','prediction_count',
                    'outliers','var_diff','bias','RMSE']

miss = 'full'
ytd_length = [np.nan,0,31,59,90,120,151,181,212,243,273,304,334]
correlations = pd.read_pickle('correlation_matrix.pickle')
metadata = pd.read_csv('metadata/md_current.csv')


with open('all_stations.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    all_stations = list(reader)


begin = time.time()
done = []

total_stations = len(all_stations)
#Initialize parameters
run_name = 'comments'
version = 'v9'
testing = True              #if testing, randomly removes some data to estimate preciction error
if miss == 'full':
    testing = False
missing_fraction = miss     #Fraction of data to remove when testing
window_size = 10             #Number of previous days to consider
#variable = var      #Climate variable to consider
coverage_req = 0.8          #Fraction of missing observations in the target station that must be covered by a valid neighbor station
corr_threshhold = 0.7       #Correlation value required between target measurements and neighbor measurements
max_neighbors = 10          #Maximum number of correlated stations that will be loaded
flatline_max = 20           #Maximum consecutive flatline imputations allowed in neighbors
hidden_layer_sizes = (100,)
max_iter = 500
alpha = 0.0001
activation = 'relu'
warm_start = False
solver = 'adam'
poss_threshhold = {'max_temp':100,'min_temp':100,'one_day_precipitation':2000}
d = pd.Timedelta('1 days')

index_list = []
for net,nid in all_stations:
    for variable in ['max_temp','min_temp','one_day_precipitation']:
        index_list.append((net,nid,variable))
idx = pd.MultiIndex.from_tuples(index_list,names=['network','station','variable'])
stat_frame = pd.DataFrame(index=idx,columns=output_statistics,data=None)
stat_frame.sort_index(inplace=True)


np.random.shuffle(all_stations)
params = {'run_name':run_name,'testing':testing,
        'missing_fraction':missing_fraction,'window_size':window_size,
        'coverage_req':coverage_req, 'corr_threshhold':corr_threshhold,
        'max_neighbors':max_neighbors, 'flatline_max':flatline_max,
        'version':version,'hidden_layer_sizes':hidden_layer_sizes,
        'max_iter':max_iter,'alpha':alpha}
with open('model_stats/test_stats/parameters_' + run_name +'.csv', 'w') as myfile:
    wr = csv.DictWriter(myfile, params.keys(),quoting=csv.QUOTE_ALL)
    wr.writeheader()
    wr.writerow(params)

while len(all_stations) > 0:
    network,nid = all_stations.pop()

    for variable in ['min_temp','max_temp','one_day_precipitation']:
        if variable == 'one_day_precipitation':
            qc_level = 8
        else:
            qc_level = 3
        result_path = 'results/' + variable + '/' + network + '_' + nid + run_name + '.pickle'

        start_time = time.time()
        failed = False
        stats = pd.DataFrame(index=['value'],columns=output_statistics,data=None)

        #Define target station
        try:
            target = pickle.load(open('../better_station_data/'+network+'/'+nid+'.pickle','rb'))
        except FileNotFoundError:
            stats['total_time'] = time.time() - start_time
            stats['result'] = 'bad record'
            target.stats = stats
            stat_frame.loc[network,nid] = stats
            pickle.dump(target, open(result_path, 'wb'))
            continue
        try:
            test = target.neighbors
        except AttributeError:
            stats['total_time'] = time.time() - start_time
            stats['result'] = 'bad object'
            target.stats = stats
            stat_frame.loc[network,nid] = stats
            pickle.dump(target, open(result_path, 'wb'))
            continue


        #Format data
        if variable not in target.data.columns:
            stats['result'] = 'missing var'
            stats['total_time'] = time.time() - start_time
            target.stats = stats
            stat_frame.loc[(network,nid,variable)] = stats.loc['value',:]
            pickle.dump(target, open(result_path, 'wb'))
            continue

        stats['variable'] = variable

        removed_dates = []
        removed_vals = []
        prediction_fail = 0
        ti = target.data.index[0]
        tf = target.data.index[-1]
        data = target.data.loc[ti:tf,variable]
        try:
            data = data.to_frame()
        except AttributeError:
            pass
        data = data.astype('float64')
        data['flag'] = 'M' #Data flagged as measured, predicted, removed, or outlier
        cols = data.columns.tolist()
        cols = [cols[1],cols[0]]
        data = data[cols]       #Reorder columns for convenience
        for row in data.index:     #Check for leading or trailing nans
            if not np.isnan(data.at[row,variable]): #(type(data.at[row,variable]) != None ) or
                ti = row
                break
        for row in reversed(data.index):
            if not np.isnan(data.at[row,variable]):
                tf = row
                break

        slice_index = pd.date_range(ti,tf,freq='D')
        if len(slice_index) < 400:
            stats['result'] = 'short record'
            stats['total_time'] = time.time() - start_time
            target.stats = stats
            stat_frame.loc[(network,nid,variable)] = stats.loc['value',:]
            pickle.dump(target, open(result_path, 'wb'))
            continue
        data = data.loc[ti:tf,:]

        name = target.md['station_name']
        try:
            stats['station_name'] = name
        except AttributeError:
            continue
        stats['record_length'] = data.shape[0]


        #Search for eligible neighbor stations
        try:
            valid_neighbors,neighbor_ids = load_correlated_gapless(network,nid,variable,corr_threshhold)
        except KeyError:
            stats['result'] = 'no anomaly'
            stats['total_time'] = time.time() - start_time
            target.stats = stats
            stat_frame.loc[(network,nid,variable)] = stats.loc['value',:]
            pickle.dump(target, open(result_path, 'wb'))
            print(5)
            continue
        if len(valid_neighbors) > max_neighbors:
            valid_neighbors = sorted(valid_neighbors,key=take_second)[:max_neighbors]

        #Select neighbors with overlapping records and combine target and neighbors into a single array
        load_time = time.time()
        reasons = []
        co = 0
        fail_counter = 0 #Number of invalid neighbors
        n_names = [] #Names of selected neighbor stations
        neighbor_fill = 0
        out_neighbors = []

        for i,neighbor_record in enumerate(valid_neighbors):
            fill_count = 0 #Number of values in neighbor station filled in with jd average
            flatline_count = 0
            n_failed = False

            time_slice = neighbor_record[0].loc[ti:tf,variable]
            time_slice = time_slice.to_frame()

            if time_slice.shape[0] / slice_index.size > coverage_req and False in np.isnan(time_slice[variable].values):
                n_normal = pd.read_pickle('../new_yearly_normals/' + neighbor_record[2] + '/' + neighbor_record[3] + '.pickle')
                n_nan_count = 0 #Counts days where neighbor and target are both NaN
                time_slice = add_nan(time_slice,slice_index)  #Add NaN values if station failed add_nan script

                #Replace NaN values in neighbor with normal for that day
                this_mean = time_slice[variable].mean(skipna=True)
                for row in time_slice.index:
                    if np.isnan(time_slice.at[row,variable]):
                        n_nan_count += 1
                        if n_nan_count >= 1000:
                            n_failed = True
                            break
                        jd = get_jd(row)
                        try:
                            if np.isnan(n_normal.at[jd,variable]):
                                if row != ti:
                                    time_slice.at[row,variable] = time_slice.at[row-d,variable]
                                else:
                                    time_slice.at[row,variable] = 0
                                flatline_count += 1
                                if flatline_count >= flatline_max:
                                    n_failed = True
                                    break
                            else:
                                time_slice.at[row,variable] = n_normal.at[jd,variable]
                                flatline_count = 0
                        except KeyError:
                            if row != ti:
                                time_slice.at[row,variable] = time_slice.at[row-d,variable]
                            else:
                                time_slice.at[row,variable] = 0
                            flatline_count += 1
                            if flatline_count >= flatline_max:
                                n_failed = True
                                break
                        fill_count += 1
                if n_failed:
                    if flatline_count >= flatline_max:
                        reasons.append('Flatline')
                    if n_nan_count >= 1000:
                        reasons.append('Too many MVs')
                    fail_counter += 1
                else:
                    co += 1
                    neighbor_fill += fill_count
                    data['n{}'.format(co)] = pd.Series(data=time_slice[variable].values,index=slice_index)
                    n_names.append((name_lookup((neighbor_record[2],neighbor_record[3])),neighbor_record[2]))
                    reasons.append('Good')
                    out_neighbors.append((neighbor_record[-2],neighbor_record[-1],n_names[-1]))
            else:
                reasons.append('Record too short')
                fail_counter += 1

        target.neighbors = out_neighbors
        stats['neighbors'] = len(valid_neighbors)-fail_counter
        stats['neighbor_fill_average'] = neighbor_fill / (co+1)
        if len(valid_neighbors) == fail_counter:
            no_neighbors = True
            stats['result'] = 'neighbors'
            failed = True
            '''
            stat_frame.at[(network,nid),'result'] = 'neighbors'
            stat_frame.at[(network,nid),'total_time'] = time.time() - start_time
            target.stats = stats
            pickle.dump(target, open(result_path, 'wb'))
            #print('insufficient neighbors')
            continue
            '''
        else:
            no_neighbors = False
            co = 0

        data.rename(columns={variable:'observation'},inplace=True)

        neighbor_time = time.time()

        #Add window of previous days' observations to each datapoint
        data,outliers = add_window(data,testing)

        propagate_time = time.time()

        #Slice  data
        nanless_data = data.copy()
        nanless_data,fraction_missing = drop_missing(nanless_data)
        stats['missing_fraction'] = fraction_missing
        true_mean = nanless_data['observation'].mean(skipna=True)
        true_var = nanless_data['observation'].var(skipna=True)

        xff = data.copy()
        if no_neighbors:
            xtrain = nanless_data.loc[:,'p0':'p{}'.format(window_size-1)]
        else:
            xtrain = nanless_data.loc[:,'n1':'p{}'.format(window_size-1)]
        ytrain = nanless_data.loc[:,'observation']
        target_std = ytrain.std()
        xff['error'] = np.nan


        slice_time = time.time()

        #Train the model
        if xtrain.shape[0] < 365:
            stats['result'] = 'insufficient training data'
            stats['total_time'] = time.time() - start_time
            target.stats = stats
            stat_frame.loc[(network,nid,variable)] = stats.loc['value',:]
            target.neighbors = out_neighbors
            pickle.dump(target, open(result_path, 'wb'))
            continue
        elif xtrain.shape[0] < 1000:
            solver = 'lbfgs'
        else:
            solver = 'adam'
        net = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,solver=solver,max_iter=max_iter,alpha=alpha,activation=activation,warm_start=warm_start)
        try:
            model = net.fit(xtrain,ytrain)
        except ValueError:
            stats['result'] = 'empty array'
            stats['total_time'] = time.time() - start_time
            target.stats = stats
            stat_frame.loc[(network,nid,variable)] = stats.loc['value',:]
            target.neighbors = out_neighbors
            pickle.dump(target, open(result_path, 'wb'))
            continue
        fit_time = time.time()


        #Forward Prediction
        total_sqerror = 0 #Stores cumulative squared error of prediction when testing on artificially missing data
        avg = [] #Stores average prediction for each day
        nan_dates = [] #Stores dates of values predicted by network
        for i,row in enumerate(xff.index):
            cell = xff.at[row,'observation']
            xff.at[row,'final'] = cell
            if no_neighbors:
                model_input = xff.loc[row,'p0':'p{}'.format(window_size-1)].values.reshape(1,-1)
            else:
                model_input = xff.loc[row,'n1':'p{}'.format(window_size-1)].values.reshape(1,-1)
            try:
                next = float(model.predict(model_input))
            except (ValueError,TypeError):
                prediction_fail += 1
                if no_neighbors:
                    next = xff.at[row,'p0']
                else:
                    next = xff.at[row,'n1']
                xff.at[row,'flag'] = 'F'
            xff.at[row,'prediction'] = next
            if variable == 'one_day_precipitation' and next < 0:
                next = 0
            if xff.at[row,'flag'] == 'M':
                error = next - xff.at[row,'observation']
                xff.at[row,'error'] = error
                total_sqerror += error
            if abs(next) > poss_threshhold[variable]:
                stats['total_time'] = time.time() - start_time
                stats['result'] = 'impossible prediction'
                xff.at[row,'flag'] = 'F'
                failed = True
                break
            if xff.at[row,'flag'] in ['R','O','X'] or np.isnan(xff.at[row,'observation']):
                if xff.at[row,'flag'] not in ['R','O']:
                    xff.at[row,'flag'] = 'P'
                xff.at[row,'final'] = next

                for offset in range(window_size):   #Diagonally propogate values into previous day columns
                    xff.at[row+(1+offset)*d,'p{}'.format(offset)] = next



                if testing:
                    if xff.at[row,'flag'] == 'R':
                        if np.isnan(error):
                            print('found nan')
                        else:
                            total_sqerror += error**2
            elif xff.at[row,'flag'] == 'F' and row < ti+10*d:
                xff.at[row,'flag'] = 'M'

            if np.isnan(xff.at[row,'final']):
                print(network)
                print(nid)
                print(variable)

        #Drop extra rows created at the end
        for i,row in enumerate(reversed(xff.index)):
            if not pd.isnull(xff.at[row,'final']):
                if row == xff.index[-1]:
                    break
                else:
                    xff.drop(index=xff.index[-i:],inplace=True)
                    break
        ff_time = time.time()

        #Fill in stat frame
        pred_var = np.var(xff['prediction'])
        pred_mean = np.mean(xff['prediction'])
        imp_var = np.var(xff['final'].loc[xff['flag'] != 'M'])
        stats['measured_var'] = true_var
        stats['predicted_var'] = pred_var
        stats['imputed_var'] = imp_var
        stats['measured_mean'] = true_mean
        stats['predicted_mean'] = pred_mean
        stats['total_time'] = time.time()-start_time
        if not failed:
            stats['result'] = 'worked'
        stats['var_diff'] = abs(true_var-imp_var)
        stats['prediction_fail'] = prediction_fail
        stats['prediction_count'] = xff.loc[xff['flag'] == 'P'].shape[0]+outliers
        stats['outliers'] = outliers
        stats['bias'] = xff['error'].mean(skipna=True)
        if variable in target.monthly_means.keys():
            if len(target.monthly_means[variable]) < 12:
                stats['outliers'] = 'failed'
        else:
            stats['outliers'] = 'failed'
        try:
            stats['RMSE'] = np.sqrt(total_sqerror / xff.loc[xff['flag']=='M'].shape[0])
        except ZeroDivisionError:
            pass


        if testing:
            try:
                stats['RMSE'] = np.sqrt(total_sqerror / len(removed_dates))
                stats['bias'] = xff['error'].mean(skipna=True)
            except ZeroDivisionError:
                stats['RMSE'] = 'BAD'
                stats['total_time'] = time.time() - start_time
                stats['result'] = 'Terrible luck'

        #Save outputs
        target.data = xff[['final','observation','prediction','error','flag']]
        target.stats = stats
        stat_frame.loc[(network,nid,variable)] = stats.loc['value',:]
        pickle.dump(target, open(result_path, 'wb'))
        done.append(network + ' ' + nid)


    if len(all_stations) % 10 == 0:
        with open('all_stations_active.csv', 'w',newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(all_stations)
        stat_frame.to_csv('stat_frame.csv')
    print(len(all_stations))

    raise SystemExit()

stat_frame.sort_values(by='result',ascending=False,inplace=True)
stat_frame.to_csv('model_stats/test_stats/stats' + run_name + '.csv')
