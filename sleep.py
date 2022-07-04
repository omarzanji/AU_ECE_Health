'''
Sleep prediction using Sleep as Android data.

author: Omar Barazanji
date: 1/16/2022
'''

from turtle import color
from matplotlib import pyplot as plt
from matplotlib import gridspec
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
                

        # map chunks of events to each timestamp and save to events_by_time
        curr_id = 0
        self.events_by_time = []
        events_s = []
        for ndx,events in enumerate(self.session_events):
            session_time_events = []
            for event in events:
                temp = curr_id
                curr_id = event.split('-')[1]
                if temp == 0: # first event
                    events_s.append(event)
                elif temp == curr_id: # add to chunk
                    events_s.append(event)
                elif ndx+1 == len(self.session_events): # last chunk of events
                    session_time_events.append(events_s)
                else:
                    events_s = []
                    session_time_events.append(events_s) # add chunk to timestamp
                    events_s.append(event)
            self.events_by_time.append(session_time_events) # append chunks to events_by_time


        # create plotable event arrays stored in event_dict
        events = self.events_by_time
        self.session_event_dict = dict()
        for sess_ndx,sess_events in enumerate(events):
            self.event_dict = dict()
            for ndx,event_chunk in enumerate(sess_events):
                for event in event_chunk:
                    name = event.split('-')[0]
                    if '_START' in name:
                        try:
                            self.event_dict[name].append((1, ndx)) # save event value and time ndx
                        except:
                            self.event_dict[name] = []
                            self.event_dict[name].append((1, ndx))
                    elif '_END' in name:
                        try:
                            self.event_dict[name].append((0, ndx)) # save event value and time ndx
                        except:
                            self.event_dict[name] = []
                            self.event_dict[name].append((0, ndx))
                    else:
                        try:
                            self.event_dict[name].append((0, ndx)) # save event value and time ndx
                        except:
                            self.event_dict[name] = []
                            self.event_dict[name].append((0, ndx))

            self.session_event_dict[sess_ndx] = self.event_dict


    def process_sessions(self):
        total_cnt = np.array(self.session_actigraphy).shape[0]
        print(f'processing {total_cnt} sessions')
        self.session_data = []
        for session_num in range(total_cnt):
            self.xplt = self.session_times[session_num]
            self.yplt = self.session_actigraphy[session_num]    

            self.light_arr = np.zeros_like(self.events_by_time[session_num])
            self.deep_arr = np.zeros_like(self.events_by_time[session_num])
            self.rem_arr = np.zeros_like(self.events_by_time[session_num])
            try:
                for ndx,vals in enumerate(self.session_event_dict[session_num]['LIGHT_START']):
                    start_ndx = vals[1]
                    end_ndx = self.session_event_dict[session_num]['LIGHT_END'][ndx][1]
                    self.light_arr[start_ndx:end_ndx+1] = 1
            except:
                pass
            try:
                for ndx,vals in enumerate(self.session_event_dict[session_num]['DEEP_START']):
                    start_ndx = vals[1]
                    end_ndx = self.session_event_dict[session_num]['DEEP_END'][ndx][1]
                    self.deep_arr[start_ndx:end_ndx+1] = 1
            except:
                pass
            try:
                for ndx,vals in enumerate(self.session_event_dict[session_num]['REM_START']):
                    start_ndx = vals[1]
                    end_ndx = self.session_event_dict[session_num]['REM_END'][ndx][1]
                    self.rem_arr[start_ndx:end_ndx+1] = 1
            except:
                pass
            self.session_data.append([self.yplt, self.light_arr, self.deep_arr, self.rem_arr])
        
        # CREATE TIME-SERIES DATA STRUCTURE WITH WINDOW SIZE
        

    def plot_sleep_session(self, session_num):
        self.xplt = self.session_times[session_num]
        self.yplt = self.session_actigraphy[session_num]    

        self.light_arr = np.zeros_like(self.events_by_time[session_num])
        self.deep_arr = np.zeros_like(self.events_by_time[session_num])
        self.rem_arr = np.zeros_like(self.events_by_time[session_num])

        for ndx,vals in enumerate(self.session_event_dict[session_num]['LIGHT_START']):
            start_ndx = vals[1]
            end_ndx = self.session_event_dict[session_num]['LIGHT_END'][ndx][1]
            self.light_arr[start_ndx:end_ndx+1] = 1
        for ndx,vals in enumerate(self.session_event_dict[session_num]['DEEP_START']):
            start_ndx = vals[1]
            end_ndx = self.session_event_dict[session_num]['DEEP_END'][ndx][1]
            self.deep_arr[start_ndx:end_ndx+1] = 1
        for ndx,vals in enumerate(self.session_event_dict[session_num]['REM_START']):
            start_ndx = vals[1]
            end_ndx = self.session_event_dict[session_num]['REM_END'][ndx][1]
            self.rem_arr[start_ndx:end_ndx+1] = 1

        fig = plt.figure()
        gs = gridspec.GridSpec(4, 1, height_ratios=[4, 2, 2, 2]) 

        ax0 = plt.subplot(gs[0])
        line0, = ax0.plot(self.xplt, self.yplt, color='b')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.ylabel('Actigraphy')
        plt.ylabel('Rate')
        plt.title('Sleep Session %d' % session_num)

        offset = len(self.events_by_time[session_num]) - len(self.xplt)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        line1, = ax1.plot(self.xplt, self.light_arr[offset:], color='g')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(gs[2], sharex=ax0, sharey=ax1)
        line2, = ax2.plot(self.xplt, self.deep_arr[offset:], color='orange')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(gs[3], sharex=ax0, sharey=ax1)
        line3, = ax3.plot(self.xplt, self.rem_arr[offset:], color='r')
                
        plt.subplots_adjust(hspace=0.1)
        plt.xlabel('Timestamp')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(self.xplt), 5))

        ax0.legend((line0,), ('actigraphy',), loc='upper right')
        ax1.legend((line1,), ('light-sleep',), loc='upper left')
        ax2.legend((line2,), ('deep-sleep',), loc='upper left')
        ax3.legend((line3,), ('rem-sleep',), loc='upper left')
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
    # sleep_session.plot_sleep_session(10)
    sleep_session.process_sessions()
    # sleep_session.export_data()