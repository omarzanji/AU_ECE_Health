"""
Processes sleep data in the format of standard sleep study or AASM standard label set.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow import keras

class AASM:

    def __init__(self):
        self.load_data()
        self.domain = 'AASM'
        model = f'../models/{self.domain}.model'
        print('Loading model...')
        self.model = keras.models.load_model(model)
        print('Done.')
        self.step_size = 5
        self.seq = 10

    def load_data(self):
        try:
            with open('../data/dodh_sessions.pkl', 'rb') as f:
                self.sessions = pickle.load(f)
        except BaseException as e:
            print(e)
            self.sessions = []

        self.x = []
        self.y = []
        if not self.sessions == []:
            for session in self.sessions.keys():
                data = []
                session_dict = self.sessions[session]
                for data_key in session_dict.keys():
                    if data_key == 'labels':
                        labels = session_dict[data_key]
                    else:
                        data.append(session_dict[data_key])
                self.x.append(data)
                self.y.append(labels)


    def create_time_series(self):
        print('\n[Creating Time-Series Data...]')
        seq = self.seq
        self.x_time_series = []
        self.y_time_series = []
        eeg_size = len(self.x[0][0])
        step_size = self.step_size
        for ndx,session in enumerate(self.x):
            if ndx==3: break # load in 12 sessions
            labels = self.y[ndx]
            for i in range(0, eeg_size, step_size):
                onehot = [0,0,0,0,0]
                try:
                    label = int(labels[i+seq])
                except IndexError:
                    break
                onehot[label] = 1
                data = []
                for sig_num, session_data in enumerate(session):
                    series = session_data[i : i+seq]
                    data.append(series)  
                self.x_time_series.append(data)
                self.y_time_series.append(onehot)

        self.x_time_series = np.array(self.x_time_series)
        self.y_time_series = np.array(self.y_time_series)

        # SAVE AS NPY CACHE FOR MAIN SLEEPNET
        domain = self.domain
        x_cache_str = f'type{domain}_SEQ{seq}_x.npy'
        y_cache_str = f'type{domain}_SEQ{seq}_y.npy'

        print('\n[Saving X and Y as cache...]')
        np.save('../cache/'+x_cache_str, self.x_time_series)
        np.save('../cache/'+y_cache_str, self.y_time_series)


    def visualize_data(self, session_num=3): # visualize testing session #13
        y = self.y[session_num]
        plt.plot(y, color='b')
        session_num+=1
        plt.title(f'Session {session_num} Sleep Scoring Ground Truth.')
        ax = plt.gca()
        ax.invert_yaxis()
        plt.ylabel(f'sleep score')
        plt.yticks([0,1,2,3,4], ['W', 'N1', 'N2', 'N3', 'R'])
        plt.show()
            

    def predict_session(self, session_num=3):
        seq = self.seq
        step_size = self.step_size
        xtest_time_series = []
        ytest_time_series = []
        session = self.x[session_num]
        labels = self.y[session_num]
        data_size = len(session[0])
        for i in range(0, data_size, step_size):
            onehot = [0,0,0,0,0]
            try:
                label = int(labels[i+seq])
            except IndexError:
                break
            onehot[label] = 1
            data = []
            for sig_num, session_data in enumerate(session):
                series = session_data[i : i+seq]
                data.append(series)  
            xtest_time_series.append(data)
            ytest_time_series.append(label)
        
        self.xtest_time_series = np.array(xtest_time_series)
        self.ytest_time_series = np.array(ytest_time_series)

        self.y_pred = []
        size = len(self.ytest_time_series)
        for itr,time_chunk in enumerate(xtest_time_series):
            input = np.expand_dims(time_chunk, axis=0) # because expected shape=(None, 16, 10)
            pred = self.model.predict(input)
            self.y_pred.append(np.argmax(pred))
            progress = (itr/size)*100
            if itr%100==0: print(f'{progress}%')
            if progress >= 50: break # stop at 50% of sleep session
        print('100.0%')

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])             
        
        ax0 = plt.subplot(gs[0])
        ax0.invert_yaxis()
        line0, = ax0.plot(ytest_time_series, color='b')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.ylabel(f'Ground Truth')
        plt.yticks([0,1,2,3,4], ['W', 'N1', 'N2', 'N3', 'R'])
        plt.title(f'Session {session_num+1} Sleep Scoring Predictions and Ground Truth.')

        ax1 = plt.subplot(gs[1])
        line1, = ax1.plot(self.y_pred, color='r')
        ax1.invert_yaxis()
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel(f'SleepNet Prediction')
        plt.yticks([0,1,2,3,4], ['W', 'N1', 'N2', 'N3', 'R'])
        
        plt.subplots_adjust(hspace=0.1)

        plt.show()


if __name__ == "__main__":
    aasm = AASM()
    aasm.create_time_series()
    # aasm.visualize_data()
    # aasm.predict_session()
