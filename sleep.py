'''
Sleep prediction using Sleep as Android data.

author: Omar Barazanji
date: 1/16/2022
'''

from matplotlib import pyplot as plt
import numpy as np
import csv

# Sleep as Android export path.
SLEEP_EXPORT_PATH = 'sleep-export.csv'

class Sleep:

    def __init__(self, data_dir):
        self.data = []
        with open(data_dir) as f:
            for line in csv.reader(f):
                self.data.append(line)

        self.x = [] # holds sleep session #'s x data (actigraphy, rem, and other raw values)
        self.y = [] # holds sleep session #'s y data (time, event, and other labels)
        for i in range(0, len(self.data) - 1, 2):
            self.y.append(self.data[i])
            self.x.append(self.data[i+1])

        self.session_events = [] # holds sleep session #'s event data
        for sess_ndx,x in enumerate(self.y):
            events = []
            # print(x)
            for label_ndx,vals in enumerate(x):
                if 'Event' in vals:
                    events.append(self.x[sess_ndx][label_ndx])
            self.session_events.append(events)
        # print(self.session_events[8])

    def plot_sleep_session(self, session_num):
        y = []
        for ndx,i in enumerate(self.y[session_num]):
            if ':' in i: 
                y.append(i)
                end_ndx = ndx

        x = np.array(self.x[session_num][15:end_ndx+1]).astype(float)
        plt.xlabel('Time')
        plt.ylabel('Rate')
        plt.title('Sleep Session %d' % session_num)
        plt.xticks(np.arange(0, len(y), 5))
        plt.plot(y, x)
        plt.show()

    def export_data(self):
        data = dict()
        data['x'] = []
        data['y'] = []
        for session_ndx,session in enumerate(self.y):
            print(session_ndx)
            x = []
            for ndx,i in enumerate(session):
                if ':' in i: 
                    x.append(i)
                    end_ndx = ndx
            try:
                y = np.array(self.x[session_ndx][15:end_ndx+1]).astype(float)
            except ValueError:
                continue
            data['x'].append(x)
            data['y'].append(y.tolist())
        import json
        with open('data.json' ,'w') as f:
            json.dump(data, f)


            

if __name__ == "__main__":

    sleep_session = Sleep(SLEEP_EXPORT_PATH)
    sleep_session.plot_sleep_session(9)
    sleep_session.export_data()