'''
Predicts sleep / awake from actigraphy data using time-series forecasting.

author: Omar Barazanji
date: 3/21/2022
organizaion: Auburn University ECE
'''

import json 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

SEQ = 10

class SleepWake:
    """
    Creates and trains LSTM sleep / wake prediction models and plots results. 
    Model is saved to models/ as a Tensorflow / Keras .model file. Data 
    is cached with type=0 (training data) and type=1 (validation data). 
    """

    def __init__(self, model=''):
        if not model == '':
            self.model = keras.models.load_model(model)


    def load_data(self, train=0):
        x_cache_str = f'type{train}_SEQ{SEQ}_x.npy'
        y_cache_str = f'type{train}_SEQ{SEQ}_y.npy'
        self.x = []
        self.y = []
        if x_cache_str in os.listdir('cache'):
            print('\n[Found cached processed data! Loading...]')
            self.x = np.load('cache/'+x_cache_str)
            self.y = np.load('cache/'+y_cache_str)
        if train:
            print('\n[Loading training data...]')
            with open('data/actigraphy.json') as f:
                self.raw_data = json.load(f)
        else: 
            print('\n[Loading validation data...]')
            with open('data/validation_actigraphy.json') as f:
                self.raw_data = json.load(f)

        # key = '5001' # add more keys from dataset later...
        # [pid, day, time_hr, sleep_time, lux, sleep_status, axis1, axis2, axis3]
        self.subjects = []

        self.day_arr = []
        self.time_hr_arr = []
        self.sleep_time_arr = []
        self.lux_arr = []
        self.sleep_status_arr = []
        self.axis1_arr = []
        self.axis2_arr = []
        self.axis3_arr = []
        self.subject_arr = []
        for ndx,key in enumerate(self.raw_data.keys()):
            self.subjects.append(key)
            for sample in self.raw_data[key]:
                self.subject_arr.append(ndx)
                self.day_arr.append(sample[1])
                self.time_hr_arr.append(sample[2])
                self.sleep_time_arr.append(str(sample[3]))
                self.lux_arr.append(sample[4])
                self.sleep_status_arr.append(sample[5])
                self.axis1_arr.append(sample[6])
                self.axis2_arr.append(sample[7])
                self.axis3_arr.append(sample[8])
        print('[Data loaded.]')
        if train:
            print(f'Training / Testing Subjects: {self.subjects}\n')
        else: print(f'Validation Subjects: {self.subjects}\n')

        if len(self.x)==0: self.process_data(train) # No cache found, create X and Y
        

    def process_data(self, train=0):
        x_cache_str = f'type{train}_SEQ{SEQ}_x.npy'
        y_cache_str = f'type{train}_SEQ{SEQ}_y.npy'

        print('\n[Creating Time-Series X and Y arrays...]\n')
        
        # STEP = 2
        self.x = []
        self.y = []
        for ndx,sample in enumerate(self.sleep_time_arr):
            try:
                axis1_series = self.axis1_arr[ndx : ndx+SEQ]
                axis2_series = self.axis2_arr[ndx : ndx+SEQ]
                axis3_series = self.axis3_arr[ndx : ndx+SEQ]
                series = [axis1_series, axis2_series, axis3_series]
                label = self.sleep_status_arr[ndx+SEQ]
                self.x.append(series)
                self.y.append(label)
            except IndexError:
                break
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        print('\n[Saving X and Y as cache...]')
        np.save('cache/'+x_cache_str, self.x)
        np.save('cache/'+y_cache_str, self.y)


    def create_model(self, units=256):
        """
        Create LSTM model with relu activation and MSE loss.
        """
        model = Sequential()
        model.add(LSTM(units, input_shape=(3,SEQ)))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('relu'))
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics='accuracy')
        return model


    def train_model(self, model, epochs=5):
        """
        Train model with optimal parameters.
        params: 
            save: 1 to save, 0 to just run.
        """
        print('\n[Training Model...]\n\n')
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y)
        self.hist = model.fit(xtrain, ytrain, epochs=epochs, verbose=1)
        self.plot_history(self.hist)
        model.summary()
        self.ypreds = model.predict(xtest)
        accuracy = accuracy_score(ytest, self.ypreds.round())
        print(f'\nAccuracy: {accuracy}\n')
        model.save('models/sleepwake.model')


    def train_model_sweep(self):
        """
        Test different model parameters to know how to fine-tune accuracy.
        """
        print('\n[Starting Parameter Sweep...]\n\n')

        units_ = [64, 128, 256, 512]
        epochs_ = [2, 5, 10, 15]
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y)

        sweep_dict = {}
        for i in range(len(units_)):
            for w in range(len(epochs_)):
                units,epochs = units_[i], epochs_[w]
                print(f'\n[Training with {units} units and {epochs} epoch(s)]\n')
                model = self.create_model(units)
                self.hist = model.fit(xtrain, ytrain, epochs=epochs, verbose=1)
                self.ypreds = model.predict(xtest)
                accuracy = accuracy_score(ytest, self.ypreds.round())
                sweep_dict[str((units,epochs))] = [self.hist, accuracy]

        keys = sweep_dict.keys()
        self.sweep_dict = {}
        for key in keys:
            hist = sweep_dict[key][0].history
            self.sweep_dict[str(key)] = {"loss": hist['loss'], "accuracy": hist['accuracy']}
        with open('param_sweep.json', 'w') as f:
            json.dump(sweep_dict, f)

    def plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper right')

        # plt.figure()
        # plt.plot(history.history['accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train'], loc='upper left')

    def generate(self, model, subject=0, day=0):
        print('\n[Generating...]')
        
        ndx_vals = []
        for ndx,sample_day in enumerate(self.day_arr):
            if subject == self.subject_arr[ndx] and sample_day == day:
                ndx_vals.append(ndx)
        start = ndx_vals[0]
        end = ndx_vals[-1]

        axis1=self.axis1_arr[start:end+1]
        axis2=self.axis2_arr[start:end+1]
        axis3=self.axis3_arr[start:end+1]
        test_x = [axis1,axis2,axis3]
        test_y_complete = self.sleep_status_arr[start:end+1]

        test_y_input = test_y_complete[0:SEQ]
        result = []

        for x in range(len(test_y_complete) - SEQ):
            axis1_series = axis1[x : x+SEQ]
            axis2_series = axis2[x : x+SEQ]
            axis3_series = axis3[x : x+SEQ]
            series = np.array([axis1_series, axis2_series, axis3_series])
            # print(series.shape)
            inp = series
            inp = inp.reshape((1, 3, SEQ))
            y = model.predict(inp, verbose=0)
            test_y_input = np.append(test_y_input, np.rint(y))

        x = np.arange(0,len(test_y_complete))
        subject_number = self.subjects[subject]

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])

        ax0 = plt.subplot(gs[0])
        line0, = ax0.plot(x[:SEQ], test_y_input[:SEQ], label='input', color='red')
        line1, = ax0.plot(x[SEQ:], test_y_input[SEQ:], label='pred', color='blue')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.title('Sleep Status Prediction | Subject %s | Day %d' % (subject_number, day))

        ax1 = plt.subplot(gs[1], sharex=ax0, sharey=ax0)
        line2, = ax1.plot(x, test_y_complete, label='actual', color='green')

        ax0.legend((line0, line1), ('input', 'predicted'), loc='upper right')
        ax1.legend((line2,), ('actual',), loc='upper right')
        plt.show()


    def visualize_data(self, subject=0, day=0):

        ndx_vals = []
        for ndx,sample_day in enumerate(self.day_arr):
            if subject == self.subject_arr[ndx] and sample_day == day:
                ndx_vals.append(ndx)
        start = ndx_vals[0]
        end = ndx_vals[-1]

        subject_number = self.subjects[subject]

        fig = plt.figure()
        gs = gridspec.GridSpec(5, 1, height_ratios=[4, 2, 2, 1, 3]) 

        ax0 = plt.subplot(gs[0])
        line0, = ax0.plot(self.sleep_time_arr[start:end+1], self.axis1_arr[start:end+1], color='r')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.title('Sleep Actigraphy Data | Subject %s | Day %d' % (subject_number, day)) 

        ax1 = plt.subplot(gs[1], sharex=ax0, sharey=ax0)
        line1, = ax1.plot(self.sleep_time_arr[start:end+1], self.axis2_arr[start:end+1], color='g')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel('Actigraphy')

        ax2 = plt.subplot(gs[2], sharex=ax0, sharey=ax0)
        line2, = ax2.plot(self.sleep_time_arr[start:end+1], self.axis3_arr[start:end+1], color='b')
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        ax3 = plt.subplot(gs[3], sharex=ax0)
        self.actigraph = np.array(self.axis1_arr[start:end+1]) + np.array(self.axis2_arr[start:end+1]) + np.array(self.axis3_arr[start:end+1])
        self.actigraph = (self.actigraph / np.max(self.actigraph)) * 10 
        line3, = ax3.plot(self.sleep_time_arr[start:end+1], self.actigraph, color='purple')
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.ylabel('Actigraphy')

        ax4 = plt.subplot(gs[4], sharex=ax0)
        line4, = ax4.plot(self.sleep_time_arr[start:end+1], self.sleep_status_arr[start:end+1], color='orange')
        plt.ylabel('Sleep\nStatus')
        ax0.legend((line0, line1, line2, line3), ('axis 1', 'axis 2', 'axis 3', 'sleep status'), loc='upper right')

        # plot settings 
        ax0.legend((line0, line1, line2, line4), ('axis 1', 'axis 2', 'axis 3', 'sleep status'), loc='upper right')
        ax3.legend((line3,), ('normalized actigraphy',), loc='upper right')   
        plt.subplots_adjust(hspace=.1)
        plt.xticks(np.arange(0, len(self.sleep_time_arr[start:end+1]), 100))
        plt.xlabel('Timestamp')
        # plt.show()

    def plot_param_sweep(self):
        with open('param_sweep.json', 'r') as f:
            param_sweep = json.load(f)
        plt.figure()
        for key in param_sweep.keys():
            plt.plot(param_sweep[key]['loss'], label=key+' loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('Param Sweep on Sleepwake LSTM Model')
        plt.yscale('log')
        plt.legend()

        plt.figure()
        for key in param_sweep.keys():
            plt.plot(param_sweep[key]['accuracy'], label=key+' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.title('Param Sweep on Sleepwake LSTM Model')
        plt.yscale('log')
        plt.legend()

        plt.show()
            
if __name__ == "__main__":
    TRAIN = 3 # 0 for testing / 1 for training / 2 for param sweep analysis

    if TRAIN == 1: # Train a new model
        sleepwake = SleepWake()

        sleepwake.load_data(TRAIN) # load training / testing data

        model = sleepwake.create_model()
        sleepwake.train_model(model)

        sleepwake.load_data(train=0) # load validation dataset (add 3 new subjects)

        sleepwake.visualize_data(8, 5) # vizualize expected result
        sleepwake.generate(model, 8, 5) # generate sleep / wake predictions and plot

    elif TRAIN == 0: # Test validation dataset
        sleepwake = SleepWake(model='models/sleepwake.model') # model trained on subjects ndx 0-5
        sleepwake.load_data(TRAIN)
        sleepwake.visualize_data(8,5) # visualize new subject's data
        sleepwake.generate(sleepwake.model, 8, 5) # test model's accuracy on new subject

    elif TRAIN == 2:
        sleepwake = SleepWake()
        sleepwake.load_data(train=1) # load training / testing data

        sleepwake.train_model_sweep()

    elif TRAIN == 3:
        sleepwake = SleepWake()
        sleepwake.plot_param_sweep()

    else:
        print('not a valid mode')