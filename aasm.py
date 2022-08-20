"""
Processes sleep data in the format of standard sleep study or AASM standard label set.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec

class AASM:

    def __init__(self):
        self.load_data()
        self.domain = 'AASM'

    def load_data(self):
        try:
            with open('data/dodh_sessions.pkl', 'rb') as f:
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


    def create_time_series(self, seq=10):

        self.x_time_series = []
        self.y_time_series = []
            
        for ndx,session in enumerate(self.x):
            if ndx==1: break
            labels = self.y[ndx]
            for i in range(int(len(session[0])/16)-seq):
                onehot = [0,0,0,0,0]
                label = int(labels[i+seq])
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
        np.save('cache/'+x_cache_str, self.x_time_series)
        np.save('cache/'+y_cache_str, self.y_time_series)


    def visualize_data(self, session_num=5, signal_num=12):
        x = self.x[session_num][signal_num]
        y = self.y[session_num]
        signal_name = list(self.sessions[f'session {session_num}'].keys())[signal_num]

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,1]) 
        
        ax0 = plt.subplot(gs[0])
        line0, = ax0.plot(x, color='b')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.ylabel(f'{signal_name}')
        plt.title(f'Session {session_num} Sleep Scoring with {signal_name}.')

        ax1 = plt.subplot(gs[1])
        line1, = ax1.plot(y, color='g')
        ax1.invert_yaxis()
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel(f'sleep score')

        plt.subplots_adjust(hspace=0.1)

        plt.show()

        

if __name__ == "__main__":
    aasm = AASM()
    # aasm.create_time_series()
    aasm.visualize_data()