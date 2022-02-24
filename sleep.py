'''
Sleep prediction using Sleep as Android data.

author: Omar Barazanji
date: 1/16/2022
'''

from matplotlib import pyplot as plt
import numpy as np
import csv

# Sleep as Android export path.
SLEEP_EXPORT_PATH = 'data/sleep-export.csv'

class Sleep:

    def __init__(self, data_dir):
        self.data = []
        with open(data_dir) as f:
            for line in csv.reader(f):
                self.data.append(line)

        self.y = [] # holds sleep session #'s x data (actigraphy, rem, and other raw values)
        self.x = [] # holds sleep session #'s y data (time, event, and other labels)
        for i in range(0, len(self.data) - 1, 2):
            self.x.append(self.data[i])
            self.y.append(self.data[i+1])


    def organize_by_events(self):
        self.session_events = [] # holds sleep session #'s event data
        self.session_times = []
        self.session_actigraphy = []

        for sess_ndx,x in enumerate(self.x):
            self.times = []
            for ndx,t in enumerate(self.x[sess_ndx]):
                if ':' in t: 
                    self.times.append(t)
                    end_ndx = ndx
            try:
                self.acti = np.array(self.y[sess_ndx][15:end_ndx+1]).astype(float)
            except:
                continue
            events = []
            for label_ndx,vals in enumerate(x):
                if 'Event' in vals:
                    events.append(self.y[sess_ndx][label_ndx])

            self.session_events.append(events)
            self.session_times.append(self.times)
            self.session_actigraphy.append(self.acti)
                

        curr_id = 0
        self.events_by_time = []
        events_s = []
        for ndx,events in enumerate(self.session_events):
            session_time_events = []
            for event in events:
                temp = curr_id
                curr_id = event.split('-')[1]
                if temp == 0:
                    events_s.append(event)
                elif temp == curr_id:
                    events_s.append(event)
                elif ndx+1 == len(self.session_events):
                    print('last one!')
                    session_time_events.append(events_s)
                else:
                    events_s = []
                    session_time_events.append(events_s)
                    events_s.append(event)
            self.events_by_time.append(session_time_events)



    def plot_sleep_session(self, session_num):
        self.xplt = self.session_times[session_num]
        self.yplt = self.session_actigraphy[session_num]
        plt.xlabel('Time')
        plt.ylabel('Rate')
        plt.title('Sleep Session %d' % session_num)
        plt.xticks(np.arange(0, len(self.xplt), 5))
        plt.plot(self.xplt, self.yplt)
        plt.show()


    def export_data(self):
        data = dict()
        data['x'] = []
        data['y'] = []
        for session_ndx,session in enumerate(self.x):
            x = []
            for ndx,i in enumerate(session):
                if ':' in i: 
                    x.append(i)
                    end_ndx = ndx
            try:
                y = np.array(self.y[session_ndx][15:end_ndx+1]).astype(float)
            except ValueError:
                continue
            data['x'].append(x)
            data['y'].append(y.tolist())
        import json
        with open('data/sleep_as_android_data.json' ,'w') as f:
            json.dump(data, f)

            

if __name__ == "__main__":

    sleep_session = Sleep(SLEEP_EXPORT_PATH)
    sleep_session.organize_by_events()
    # sleep_session.plot_sleep_session(9)
    # sleep_session.export_data()