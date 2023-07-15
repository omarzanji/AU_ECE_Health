"""
Processes sleep data in the format of standard sleep study or AASM standard label set.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow import keras
from scipy import signal

STFT = True


class AASM:

    def __init__(self, create_training_data=True, time_series=True):
        self.time_series = time_series
        self.create_training_data = create_training_data
        self.domain = 'AASM'
        self.step_size = 2
        self.seq = 2
        self.load_data()

    def load_model(self):
        model = f'../models/{self.domain}.model'
        print('Loading model...')
        self.model = keras.models.load_model(model)
        print('Done.')

    def load_data(self):
        print('\n[Loading data...]')
        try:

            if STFT:
                with open('../data/dodh_sessions.pkl', 'rb') as f:
                    self.sessions = pickle.load(f)
            else:
                with open('../data/dodh_sessions_no_spect.pkl', 'rb') as f:
                    self.sessions = pickle.load(f)
        except BaseException as e:
            print(e)
            self.sessions = []
        if self.create_training_data == False:
            return
        self.x = []
        self.y = []
        if not self.sessions == []:
            for session in self.sessions.keys():
                data = []
                session_dict = self.sessions[session]
                for data_key in session_dict.keys():
                    if data_key == 'labels':
                        labels = session_dict[data_key]
                        for label in labels:
                            onehot = [0, 0, 0, 0, 0]
                            onehot[label] = 1
                            self.y.append(onehot)
                    else:
                        data.append(session_dict[data_key])

                for i in range(len(labels)):
                    single_sample_data = []
                    for signal_data in data:
                        single_sample_data.append(signal_data[i])
                    self.x.append(single_sample_data)

            print('xshape: ', np.array(self.x).shape)
            print('yshape: ', np.array(self.y).shape)

        if self.time_series:
            self.create_time_series()
        else:
            domain = self.domain
            if not STFT:
                domain = 'AASM_NO_SPECT'
            x_cache_str = f'type{domain}_SEQ{16}_x.npy'
            y_cache_str = f'type{domain}_SEQ{16}_y.npy'
            np.save('../cache/'+x_cache_str, self.x)
            np.save('../cache/'+y_cache_str, self.y)

    def create_time_series(self):
        print('\n[Creating Time-Series Data...]\n')
        seq = self.seq
        self.x_time_series = []
        self.y_time_series = []
        for i in range(0, len(self.x) - self.seq, self.step_size):
            data = self.x[i: i+self.seq]
            label = self.y[i+self.seq]
            concat_spects = []  # will be of size self.seq * 16
            for time_step in data:
                for spect in time_step:
                    concat_spects.append(spect)
            self.x_time_series.append(concat_spects)
            self.y_time_series.append(label)

        print('Converting to np array...')
        self.x_time_series = np.array(self.x_time_series, dtype=np.float32)
        self.y_time_series = np.array(self.y_time_series)

        # SAVE AS NPY CACHE FOR MAIN SLEEPNET
        domain = self.domain
        x_cache_str = f'type{domain}_SEQ{seq}_x.npy'
        y_cache_str = f'type{domain}_SEQ{seq}_y.npy'

        print('\n[Saving X and Y as cache...]')
        if not os.path.exists('../cache/'):
            os.mkdir('../cache/')
        np.save('../cache/'+x_cache_str, self.x_time_series)
        np.save('../cache/'+y_cache_str, self.y_time_series)

    def visualize_data(self, session_num=5):  # visualize testing session #13
        y = self.sessions[list(self.sessions.keys())[session_num]]['labels']
        plt.plot(y, color='b')
        session_num += 1
        plt.title(f'Session {session_num} Sleep Scoring Ground Truth.')
        ax = plt.gca()
        ax.invert_yaxis()
        plt.ylabel(f'sleep score')
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])
        plt.show()

    def predict_session(self, session_num):
        self.load_model()
        seq = self.seq
        step_size = self.step_size
        xtest_time_series = []
        ytest_time_series = []

        session = []  # will hold raw session data at session_num
        session_x = []  # will hold transformed session data ready for tensor

        # get labels from self.sessions
        labels = self.sessions[list(self.sessions.keys())[
            session_num]]['labels']

        # get session data from self.sessions
        for data_key in self.sessions[list(self.sessions.keys())[
                session_num]].keys():
            if not data_key == 'labels':
                session.append(self.sessions[list(self.sessions.keys())[
                    session_num]][data_key])

        # transform to be in correct shape for training
        for i in range(len(labels)):
            single_sample_data = []
            for signal_data in session:
                single_sample_data.append(signal_data[i])
            session_x.append(single_sample_data)

        # create ground truth hypnogram time series
        # for i in range(0, len(labels) - self.seq, step_size):
        #     data = session_x[i: i+self.seq]
        #     onehot = [0, 0, 0, 0, 0]
        #     label = labels[i+self.seq]
        #     onehot[label] = 1
        #     concat_spects = []  # will be of size self.seq * 16
        #     for time_step in data:
        #         for spect in time_step:
        #             concat_spects.append(spect)
        #     xtest_time_series.append(concat_spects)
        #     ytest_time_series.append(onehot)

        self.xtest_time_series = np.array(session_x)
        self.ytest_time_series = np.array(labels)

        self.y_pred = []
        size = len(self.ytest_time_series)
        for itr, sample in enumerate(self.xtest_time_series):
            pred = self.model.predict(np.expand_dims(sample, 0))
            self.y_pred.append(np.argmax(pred))
            progress = (itr/size)*100
            if itr % 100 == 0:
                print(f'{progress}%')
            if progress >= 50:
                break  # stop at 50% of sleep session
        print('100.0%')

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        ax0 = plt.subplot(gs[0])
        ax0.invert_yaxis()
        ytest_by_class = []
        for x in self.ytest_time_series:
            ytest_by_class.append(np.argmax(x))
        line0, = ax0.plot(ytest_by_class, color='b')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.ylabel(f'Ground Truth')
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])
        plt.title(
            f'Session {session_num+1} Sleep Scoring Predictions and Ground Truth.')

        ax1 = plt.subplot(gs[1])
        line1, = ax1.plot(self.y_pred, color='r')
        ax1.invert_yaxis()
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel(f'SleepNet Prediction')
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])

        plt.subplots_adjust(hspace=0.1)

        plt.show()


if __name__ == "__main__":
    aasm = AASM(
        create_training_data=True,
        # create_training_data=False,
        time_series=False
    )
    # aasm.visualize_data(0)
    # aasm.predict_session(10)
