'''
Predicts sleep / awake from actigraphy data + a few other models.

author: Omar Barazanji
'''

import json 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from sklearn.model_selection import train_test_split
import tensorflow as tf


class SleepWake:

    def __init__(self):
        pass

    def load_data(self):
        with open('data/actigraphy.json') as f:
            self.raw_data = json.load(f)
        key = '5001' # add more keys from dataset later...
        # [pid, day, time_hr, sleep_time, lux, sleep_status, axis1, axis2, axis3]
        self.day_arr = []
        self.time_hr_arr = []
        self.sleep_time_arr = []
        self.lux_arr = []
        self.sleep_status_arr = []
        self.axis1_arr = []
        self.axis2_arr = []
        self.axis3_arr = []
        for sample in self.raw_data[key]:
            self.day_arr.append(sample[1])
            self.time_hr_arr.append(sample[2])
            self.sleep_time_arr.append(str(sample[3]))
            self.lux_arr.append(sample[4])
            self.sleep_status_arr.append(sample[5])
            self.axis1_arr.append(sample[6])
            self.axis2_arr.append(sample[7])
            self.axis3_arr.append(sample[8])


    def process_data(self):
        SEQ = 30
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


    def create_model(self):
        SEQ= 30
        model = Sequential()
        model.add(LSTM(SEQ, input_shape=(3,SEQ)))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
        return model

    def train_model(self, model):
        hist = model.fit(self.x, self.y, epochs=4, verbose=1)
        model.summary()
        model.save('models/sleepwake.model')
        return hist

    def generate(self, model, day):
        SEQ=30
        
        ndx_vals = []
        for ndx,sample in enumerate(self.day_arr):
            if sample == day:
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
            test_y_input = np.append(test_y_input, y)

        x = np.arange(0,len(test_y_complete))

        fig = plt.figure(0)
        plt.plot(x[:SEQ], test_y_input[:SEQ], label='input', color='red')
        plt.plot(x[SEQ:], test_y_input[SEQ:], label='pred', color='blue')
        # fig = plt.figure(1)
        plt.plot(x, test_y_complete, label='actual', color='green')
        plt.legend()
        plt.show()


    def visualize_data(self, day):

        ndx_vals = []
        for ndx,sample in enumerate(self.day_arr):
            if sample == day:
                ndx_vals.append(ndx)
        start = ndx_vals[0]
        end = ndx_vals[-1]

        fig = plt.figure()
        gs = gridspec.GridSpec(4, 1, height_ratios=[4, 2, 2, 1]) 

        ax0 = plt.subplot(gs[0])
        line0, = ax0.plot(self.sleep_time_arr[start:end+1], self.axis1_arr[start:end+1], color='r')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.title('Sleep Actigraphy Data | Subject 5001 | Day %d' % day) 

        ax1 = plt.subplot(gs[1], sharex=ax0)
        line1, = ax1.plot(self.sleep_time_arr[start:end+1], self.axis2_arr[start:end+1], color='g')
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(gs[2], sharex=ax0)
        line2, = ax2.plot(self.sleep_time_arr[start:end+1], self.axis3_arr[start:end+1], color='b')
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(gs[3], sharex=ax0)
        line3, = ax3.plot(self.sleep_time_arr[start:end+1], self.sleep_status_arr[start:end+1], color='orange')

        ax0.legend((line0, line1, line2, line3), ('axis 1', 'axis 2', 'axis 3', 'sleep status'), loc='top right')
                
        plt.subplots_adjust(hspace=.1)
        plt.xticks(np.arange(0, len(self.sleep_time_arr[start:end+1]), 100))
        plt.show()

if __name__ == "__main__":
    
    sleepwake = SleepWake()
    sleepwake.load_data()  
    sleepwake.process_data()
    model = sleepwake.create_model()
    sleepwake.train_model(model)
    sleepwake.generate(model, 10)