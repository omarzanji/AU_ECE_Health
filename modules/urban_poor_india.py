"""
Creates X and Y arrays in cache folder ready for time-series forecasting. 
"""

from matplotlib import pyplot as plt
from matplotlib import gridspec
from tensorflow import keras
import numpy as np
import json
import os

TRAIN = 0
SEQ = 10

class UrbanPoorIndia:

    def __init__(self, model=''):
        if not model == '':
            # load a model created from main.py (SleepNet)
            self.model = keras.models.load_model(model)

    def load_data(self, train=0, domain='UrbanPoorIndia'):
        """
        Loads data. 
        params:
            train: 1 generates new X and Y cache, 0 loads validation data.
            domain: str for naming X and Y cache.
        """
        x_cache_str = f'type{domain}_SEQ{SEQ}_x.npy'
        y_cache_str = f'type{domain}_SEQ{SEQ}_y.npy'
        self.x = []
        self.y = []
        try:
            if x_cache_str in os.listdir('../cache'):
                print('\n[Found cached processed data! Loading...]')
                self.x = np.load('../cache/'+x_cache_str)
                self.y = np.load('../cache/'+y_cache_str)
        except:
            print('Creating cache folder...')
            os.mkdir('cache')
        if train:
            print('\n[Loading training data...]')
            with open('../data/actigraphy.json') as f:
                self.raw_data = json.load(f)
        else: 
            print('\n[Loading validation data...]')
            with open('../data/validation_actigraphy.json') as f:
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
        

    def process_data(self, train=0, domain='UrbanPoorIndia'):
        x_cache_str = f'type{domain}_SEQ{SEQ}_x.npy'
        y_cache_str = f'type{domain}_SEQ{SEQ}_y.npy'

        print('\n[Creating Time-Series X and Y arrays...]\n')
        
        # NEED TO ADD VARIABLE AXIS DIM (for sleep as android: 1d actigraphy...)

        # STEP = 2
        self.x = []
        self.y = []
        for ndx,sample in enumerate(self.sleep_time_arr):
            try:
                axis1_series = self.axis1_arr[ndx : ndx+SEQ]
                axis2_series = self.axis2_arr[ndx : ndx+SEQ]
                axis3_series = self.axis3_arr[ndx : ndx+SEQ]
                series = [axis1_series, axis2_series, axis3_series]

                # NEED TO ADD VARIABLE LABEL DIM (for sleep as android: 3d labels...)
                label = self.sleep_status_arr[ndx+SEQ]
                self.x.append(series)
                self.y.append(label)
            except IndexError:
                break
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        print('\n[Saving X and Y as cache...]')
        np.save('../cache/'+x_cache_str, self.x)
        np.save('../cache/'+y_cache_str, self.y)


    def generate(self, subject=0, day=0):
        print('\n[Generating...]')
        
        model = self.model

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
        plt.yticks([1, 0], ['sleep', 'wake'])
        ax0.set_ylabel('Sleep Status')
        line0, = ax0.plot(x[:SEQ], test_y_input[:SEQ].astype(int), label='input', color='red')
        line1, = ax0.plot(x[SEQ:], test_y_input[SEQ:].astype(int), label='pred', color='blue')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.title('Sleep Status Prediction | Subject %s | Day %d' % (subject_number, day))

        ax1 = plt.subplot(gs[1], sharex=ax0, sharey=ax0)
        ax1.set_ylabel('Sleep Status')
        plt.yticks([1, 0], ['sleep', 'wake'])
        line2, = ax1.plot(x, np.array(test_y_complete).astype(int), label='actual', color='green')

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
        gs = gridspec.GridSpec(4, 1, height_ratios=[4, 2, 2, 3]) 

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
        
        # ax3 = plt.subplot(gs[3], sharex=ax0)
        # self.actigraph = np.array(self.axis1_arr[start:end+1]) + np.array(self.axis2_arr[start:end+1]) + np.array(self.axis3_arr[start:end+1])
        # self.actigraph = (self.actigraph / np.max(self.actigraph)) * 10 
        # line3, = ax3.plot(self.sleep_time_arr[start:end+1], self.actigraph, color='purple')
        # plt.setp(ax3.get_xticklabels(), visible=False)
        # plt.ylabel('Actigraphy')

        ax3 = plt.subplot(gs[3], sharex=ax0)
        plt.yticks([1,0], ['sleep', 'wake'])
        line3, = ax3.plot(self.sleep_time_arr[start:end+1], np.array(self.sleep_status_arr[start:end+1]).astype(int), color='orange')
        plt.ylabel('Sleep\nStatus')

        # plot settings 
        ax0.legend((line0, line1, line2, line3), ('axis 1', 'axis 2', 'axis 3', 'sleep status'), loc='upper right')
        plt.subplots_adjust(hspace=.1)
        plt.xticks(np.arange(0, len(self.sleep_time_arr[start:end+1]), 100))
        plt.xlabel('Timestamp')
        # plt.show()

if __name__ == "__main__":

    if TRAIN==1: # create x,y time series arrays for training new model with ../main.py
        sleep_net = UrbanPoorIndia(model='')
        sleep_net.load_data(train=TRAIN)
        sleep_net.process_data()
    else:
        sleep_net = UrbanPoorIndia(model='../models/UrbanPoorIndia.model')
        sleep_net.load_data(train=TRAIN)
        sleep_net.visualize_data(0, 5)
        sleep_net.generate(0, 5)
        plt.show()