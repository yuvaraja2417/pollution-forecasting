#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from multitcn_components import TCNStack, DownsampleLayerWithAttention, LearningRateLogger
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow_addons as tfa
import uuid
import sys
from scipy.signal import correlate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib
import matplotlib.colors as colors
from IPython.display import Image
from scipy import stats
from tqdm import tqdm
import seaborn as sns


# In[ ]:


def windowed_dataset(series, time_series_number, window_size):
    """
    Returns a windowed dataset from a Pandas dataframe
    """
    available_examples= series.shape[0]-window_size + 1
    time_series_number = series.shape[1]
    inputs = np.zeros((available_examples,window_size,time_series_number))
    for i in range(available_examples):
        inputs[i,:,:] = series[i:i+window_size,:]
    return inputs 

def windowed_forecast(series, forecast_horizon):
    available_outputs = series.shape[0]- forecast_horizon + 1
    output_series_num = series.shape[1]
    output = np.zeros((available_outputs,forecast_horizon, output_series_num))
    for i in range(available_outputs):
        output[i,:]= series[i:i+forecast_horizon,:]
    return output

def shuffle_arrays_together(a,b):
    p = np.random.permutation(a.shape[0])
    return a[p],b[p]

def norm_cross_corr(a,b):
    nom = correlate(a,b)
    den = np.sqrt(np.sum(np.power(a,2))*np.sum(np.power(b,2)))
    return nom/den

def symm_mape(true,prediction):
    return 100*np.sum(2*np.abs(prediction-true)/(np.abs(true)+np.abs(prediction)))/true.size

def get_metrics(true,prediction,print_metrics=False):
        c = norm_cross_corr(true,prediction)
        extent = int((c.shape[0]-1)/2)
        max_corr_point = np.argmax(c)-extent
        max_corr = np.max(c)
        max_v = np.max(prediction)
        mse = mean_squared_error(true,prediction,squared=True)
        rmse = mean_squared_error(true,prediction,squared=False)
        mae = mean_absolute_error(true,prediction)
        r2 = r2_score(true,prediction)
        smape = symm_mape(true,prediction)
        if print_metrics:
            print("Max %f - Autocorr %d - MSE %f - RMSE %f - MAE %f - sMAPE %f%% - R^2 %f"%(max_v,max_corr_point,mse,rmse,mae,smape,r2))
        return [max_corr_point,mse,rmse,mae,smape,r2]

def get_confidence_interval_series(sample_array,confidence_level=0.95):
    bounds = stats.t.interval(confidence_level,sample_array.shape[0]-1)
    samples_mean = np.mean(sample_array,axis=0)
    samples_std = np.std(sample_array,axis=0,ddof=1)
    lower_bound = samples_mean + bounds[0]*samples_std/np.sqrt(sample_array.shape[0])
    upper_bound = samples_mean + bounds[1]*samples_std/np.sqrt(sample_array.shape[0])
    return samples_mean, lower_bound, upper_bound

def present_mean_metrics(metrics):
    print("Autocorr\t\t MSE\t\t RMSE\t\t MAE\t\t sMAPE\t\t R^2")
    print("%10.4f\t %10.4f\t %10.4f\t %10.4f\t %10.4f\t %10.4f"% tuple(np.mean(metrics,axis=0)))
    print("+-",)
    print("%10.4f\t %10.4f\t %10.4f\t %10.4f\t %10.4f\t %10.4f"% tuple(np.std(metrics,axis=0,ddof=1)))
    


# In[ ]:


loss = 'mse'
#Dataset parameters
window_length = 96
forecast_horizon = 24
preprocessor = preprocessing.MinMaxScaler()
out_preprocessor = preprocessing.MinMaxScaler()
# preprocessor = preprocessing.StandardScaler(with_mean=0,with_std=1)
# out_preprocessor = preprocessing.StandardScaler(with_mean=0,with_std=1)
shuffle_train_set = True
scale_output = True
training_percentage = 0.75
experiment_target = F"Forecasting,{forecast_horizon} steps ahead"
experiment_complete = False


# In[ ]:


############## Set up model ##########################
class MTCNAModel(tf.keras.Model):
    
    def __init__(self, tcn_layer_num,tcn_kernel_size,tcn_filter_num,window_size,forecast_horizon,num_output_time_series, use_bias, kernel_initializer, tcn_dropout_rate,tcn_dropout_format,tcn_activation, tcn_final_activation, tcn_final_stack_activation):
        super(MTCNAModel, self).__init__()


        self.num_output_time_series = num_output_time_series
        

        #Create stack of TCN layers    
        self.lower_tcn = TCNStack(tcn_layer_num,tcn_filter_num,tcn_kernel_size,window_size,use_bias,kernel_initializer,tcn_dropout_rate,tcn_dropout_format,tcn_activation,tcn_final_activation, tcn_final_stack_activation)
        
        self.downsample_att = DownsampleLayerWithAttention(num_output_time_series,window_size, tcn_kernel_size, forecast_horizon, kernel_initializer, None)
    
        
        
    def call(self, input_tensor):
        x = self.lower_tcn(input_tensor)
        x, distribution = self.downsample_att([x,input_tensor[:,:,:self.num_output_time_series]])
        return [x[:,i,:] for i in range(self.num_output_time_series)], distribution


# In[ ]:


################ Prepare dataset ###########################

### Note details for logging purposes
dataset_description = "Italian air quality data"
dataset_preprocessing = """Drop time information, Remove NAN rows at end, Replace missing values with 0"""

data = pd.read_csv("Datasets/AirQualityUCI.csv",sep=';',decimal=',')
## Remove NaN rows
data = data.drop(np.arange(9357,9471,1))
# Remove emtpy columns
data = data.drop(['Unnamed: 15','Unnamed: 16'],axis=1)


#Create date object for easy splitting according to dates
dateobj = pd.to_datetime(data["Date"],dayfirst=True) + pd.to_timedelta(data["Time"].str.replace(".00.00",":00:00"))

### For now remove timestamp and output values
data = data.drop(columns=["Date","Time"],axis=1)

#Drop column due to high number of missing values
data = data.drop(['NMHC(GT)'],axis=1)

# Replace missing values with 0
data = data.replace(-200,0)

# Reorganize columns in preparation for second stage (first columns are in order of outputs)
columns = ['CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)','PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
data = data[columns]


## Add date object for splitting
data['DateObj'] = dateobj


# In[ ]:


#Split data based on dates
training_start_date = pd.Timestamp(year=2004,month=3,day=10)

# Preceding values used only for creating final graph and predicting first values of test set
holdout_preceding_date = pd.Timestamp(year=2004, month=11, day=11)


holdout_set_start_date = pd.Timestamp(year=2004, month=12, day=11)
holdout_set_end_date = pd.Timestamp(year=2005, month=4, day=5)

training_data = data.loc[(data['DateObj']>=training_start_date) & (data['DateObj'] < holdout_set_start_date)]
test_data = data.loc[(data['DateObj'] >= holdout_set_start_date) & (data['DateObj'] < holdout_set_end_date)]

pre_evaluation_period = data.loc[(data['DateObj'] >= holdout_preceding_date) & (data['DateObj'] < holdout_set_start_date)]

input_variables = list(training_data.columns)

training_data = training_data.drop(['DateObj'],axis=1)
test_data = test_data.drop(['DateObj'],axis=1)


# In[ ]:


##Select prediction target
targets = ['CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)']
labels = np.array(training_data[targets])


if scale_output:
    out_preprocessor.fit(labels)
    if "Normalizer" in str(out_preprocessor.__class__):
        ## Save norm so in case of normalizer we can scale the predictions correctly
        out_norm = np.linalg.norm(labels)
        labels = preprocessing.normalize(labels,axis=0)
    else:
        labels= out_preprocessor.transform(labels)


num_input_time_series = training_data.shape[1]


### Make sure data are np arrays in case we skip preprocessing
training_data = np.array(training_data)

### Fit preprocessor to training data
preprocessor.fit(training_data)

if "Normalizer" in str(preprocessor.__class__):
    ## Save norm so in case of normalizer we can scale the test_data correctly
    in_norm = np.linalg.norm(training_data,axis=0)
    training_data = preprocessing.normalize(training_data,axis=0)
else:
    training_data = preprocessor.transform(training_data)


# In[ ]:



### Create windows for all data
data_windows = windowed_dataset(training_data[:-forecast_horizon],num_input_time_series,window_length)
label_windows = windowed_forecast(labels[window_length:],forecast_horizon)

### Transpose outputs to agree with model output
label_windows = np.transpose(label_windows,[0,2,1])


samples = data_windows.shape[0]


## Shuffle windows
if shuffle_train_set:
    data_windows, label_windows = shuffle_arrays_together(data_windows,label_windows)

### Create train and validation sets
train_x = data_windows
train_y = [label_windows[:,i,:] for i in range(len(targets))]


## In order to use all days of test set for prediction, append training window from preceding period
pre_test_train = pre_evaluation_period[test_data.columns][-window_length:]
test_data = pd.concat([pre_test_train,test_data])

## Create windowed test set with same process
test_labels = np.array(test_data[targets])

#### Preprocess data
test_data = np.array(test_data)

if "Normalizer" in str(preprocessor.__class__):
    test_data = test_data/in_norm
else:
    test_data = preprocessor.transform(test_data)

test_x = windowed_dataset(test_data[:-forecast_horizon],num_input_time_series,window_length)
test_y = np.transpose(windowed_forecast(test_labels[window_length:],forecast_horizon),[0,2,1])

## Create pre test period for visualization
pre_test_target = np.append(np.array(pre_evaluation_period[targets]),test_labels[:window_length])

total_samples = train_x.shape[0] + test_x.shape[0]


# In[ ]:


##################### Initialize model parameters ########################
## For simplicity all time series TCNs have the same parameters, though it is relatively easy to change this
tcn_kernel_size = 3
tcn_layer_num = 5
tcn_use_bias = True
tcn_filter_num = 128
tcn_kernel_initializer = 'random_normal'
tcn_dropout_rate = 0.3 
tcn_dropout_format = "channel"
tcn_activation = 'relu'
tcn_final_activation = 'linear'
tcn_final_stack_activation = 'relu'
loss = [loss]*len(targets)


# In[ ]:


# ### Check for GPU

## Make only given GPU visible   
gpus = tf.config.experimental.list_physical_devices('GPU')
mirrored_strategy = None

print("GPUs Available: ", gpus)
if len(gpus)==0:
    device = "CPU:0"
else:
    print("Enter number of gpus to use:")
    gpu_num = input()
    if len(gpu_num)!=0 and gpu_num.isdigit():
        gpu_num = int(gpu_num)
    if gpu_num==1:
        print("Enter index of GPU to use:")
        gpu_idx = input()
        if len(gpu_idx)!=0 and gpu_idx.isdigit():
            gpu_idx = int(gpu_idx)
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
        device = "GPU:0"
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=[F"GPU:{i}" for i in range(gpu_num)])
        device = " ".join([F"GPU:{i}" for i in range(gpu_num)])


# In[ ]:


### Set evaluation seed to affect dropout random execution
print("Enter a seed for the evaluation:")
seed = input()
if len(seed)!=0 and seed.isdigit():
    seed = int(seed)
else:
    seed = 192
np.random.seed(seed)
tf.random.set_seed(seed)


# In[ ]:


## Set up test model
## From all the test samples keep individual, non overlapping days
test_x_days = test_x[0::forecast_horizon,:]
true_y = np.transpose(test_y[0::forecast_horizon,:],(0,2,1)).reshape((-1,len(targets)))

test_dropout = 0.3

with tf.device(device):
    test_model = MTCNAModel(tcn_layer_num,tcn_kernel_size,tcn_filter_num,window_length,forecast_horizon,len(targets), tcn_use_bias, tcn_kernel_initializer, test_dropout, tcn_dropout_format, tcn_activation, tcn_final_activation, tcn_final_stack_activation)
_ = test_model(train_x[0:1])


best_weight_name = "510e465d-c041-4fb3-b76c-f514fde218ae-weights.112-0.0282.h5"

## Generate predictions for test set using best weight (first in list)
## Reset training fase to disable dropout 
tf.keras.backend.set_learning_phase(0)
test_model.load_weights("SecondStageWeights-AirQ/"+best_weight_name)

best_pred = np.asarray(test_model(test_x_days)[0]).reshape((len(targets),-1)).T
if scale_output and "Normalizer" in str(out_preprocessor.__class__):
    best_pred *= (out_norm)
else:
    best_pred = out_preprocessor.inverse_transform(best_pred)


# In[ ]:



from os import listdir
weight_names = listdir("SecondStageWeights-AirQ/")
dropout_runs_per_weight = 20

metrics_number = 6
samples_per_prediction = dropout_runs_per_weight*len(weight_names)

## Enable dropout
tf.keras.backend.set_learning_phase(1)

dl_errors  = np.zeros((samples_per_prediction,test_x_days.shape[0]*forecast_horizon,len(targets)))
dl_predictions = np.zeros((samples_per_prediction,test_x_days.shape[0]*forecast_horizon,len(targets)))
dl_metrics = np.zeros((samples_per_prediction,metrics_number,len(targets)))

for i in tqdm(range(len(weight_names))):
    test_model.load_weights("SecondStageWeights-AirQ/"+weight_names[i])
    for j in range(dropout_runs_per_weight):
        ## Get DL test set predictions and metrics
        cur_pred = np.asarray(test_model(test_x_days)[0]).reshape((len(targets),-1)).T
        if scale_output and "Normalizer" in str(out_preprocessor.__class__):
            cur_pred *= (out_norm)
        else:
            cur_pred = out_preprocessor.inverse_transform(cur_pred)
        dl_predictions[i*dropout_runs_per_weight+j,:] = cur_pred
        dl_errors[i*dropout_runs_per_weight+j,:] = cur_pred - true_y
        for t in range(len(targets)):
            dl_metrics[i*dropout_runs_per_weight+j,:,t] = np.asarray(get_metrics(true_y[:,t],cur_pred[:,t],print_metrics=False))


# In[ ]:


np.set_printoptions(linewidth=100)
sns.set()
for var_idx in range(len(targets)):
    print(targets[var_idx])
    present_mean_metrics(dl_metrics[...,var_idx])
    
    fig = plt.figure(figsize=(20,10))
    plt.hist(dl_errors[...,var_idx].flatten(),alpha=0.5)
    plt.hist((dl_predictions[...,var_idx]-np.median(dl_predictions[...,var_idx],axis=0)).flatten(),alpha=0.5)
    plt.show()


# In[ ]:


pred_mean, dl_lower_bound, dl_upper_bound = get_confidence_interval_series(dl_predictions)


# In[ ]:


preceding_points = 24
from_day = 10
to_day = 20

pred_plot_range = range(preceding_points,preceding_points+(to_day-from_day)*forecast_horizon)
pred_sp = from_day*forecast_horizon
pred_ep = to_day*forecast_horizon

for i in range(len(targets)):
    fig = plt.figure(figsize=(20,10))
    plt.plot(pred_plot_range,pred_mean[pred_sp:pred_ep,i],marker="o",label="Prediction")
    plt.fill_between(pred_plot_range, dl_lower_bound[pred_sp:pred_ep,i], dl_upper_bound[pred_sp:pred_ep,i], alpha=0.3)
    
    if from_day==0:
        plt.plot(pre_test_target[-preceding_points:,i],label="Pretest period", marker="o")
    else:
        plt.plot(true_y[pred_sp-preceding_points:pred_sp,i],label="Pretest period", marker="o")
    plt.plot(pred_plot_range,true_y[from_day*forecast_horizon:to_day*forecast_horizon,i],marker="o",label="True data")

    plt.grid(axis='x')
    plt.legend()
    plt.title(targets[i])
    plt.show()


# In[ ]:


## Present attention graphs for specific prediction output

input_variables = ['CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)']

var_of_interest = 'C6H6(GT)'

var_idx = input_variables.index(var_of_interest)

test_idx = 45

## Reset training fase to disable dropout 
tf.keras.backend.set_learning_phase(0)
test_model.load_weights("SecondStageWeights-AirQ/"+best_weight_name)


o, dist = test_model(test_x_days[test_idx:test_idx+1])

o = np.asarray(o).reshape((len(targets),-1)).T
if scale_output:
    if "Normalizer" in str(out_preprocessor.__class__):
        o *= (out_norm)
    else:
        o = out_preprocessor.inverse_transform(o)
        
inp = preprocessor.inverse_transform(test_x_days[test_idx])[:,var_idx]

prediction= o[:,var_idx]
true_out = true_y[test_idx*forecast_horizon:(test_idx+1)*(forecast_horizon),var_idx]


# In[ ]:


fix, ax = plt.subplots(figsize=(20,10))
plt.plot(inp)
plt.plot(np.arange(window_length,window_length+forecast_horizon),prediction,marker="o",label="Prediction")
plt.plot(np.arange(window_length,window_length+forecast_horizon),true_out,marker="o",label="Ground truth")
plt.legend()
plt.show()


# In[ ]:


## Get value dense layer
for w in test_model.weights:
    if w.name.endswith("sep_dense_value_weights:0"):
        weights = np.abs(w.numpy())[var_idx]
        #weights = w.numpy()[var_idx]
        break

dist_var = dist.numpy()[0,var_idx,...]
full_dist = np.matmul(dist_var,weights.T)


# In[ ]:


sns.set()
def infl_to_out_elem(out_elem):
    elem_dist = full_dist[out_elem:out_elem+1,:]
    prep = preprocessing.MinMaxScaler()
    prep.fit(elem_dist.T)
    elem_dist = prep.transform(elem_dist.T)
        
    fig, ax = plt.subplots(figsize=(20,10))
    sns.heatmap(elem_dist.T, cmap="Blues", cbar=True, yticklabels=False, xticklabels=10)
    ax2 = plt.twinx()
    ax2.plot(range(window_length,window_length+forecast_horizon),true_out,label="Ground truth",marker="o")
    ax2.plot(range(window_length,window_length+forecast_horizon),prediction,label="Prediction",marker="o")
    plt.plot([window_length+out_elem], [prediction[out_elem]], marker='o', label= "Step "+str(out_elem+1), markersize=8, color="black")
    sns.lineplot(x=np.arange(0,window_length),y=inp, ax=ax2)
    ax.axis('tight')
    ax2.legend(fontsize=20)
    plt.show()
#     plt.savefig("dist_images/%s-%02d.png"%(var_of_interest,out_elem))
#     plt.close(fig)

interact(infl_to_out_elem, out_elem=(0,forecast_horizon-1,1))
#infl_to_out_elem(12)



# In[ ]:





# In[ ]:




