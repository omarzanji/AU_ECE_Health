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

        # self.x = self.data[2][15:75]
        # self.y = np.array(self.data[3][15:75]).astype(float)

        self.x = []
        self.y = []
        for i in range(0, len(self.data) - 1, 2):
            self.y.append(self.data[i])
            self.x.append(self.data[i+1])


    def plot_sleep_session(self, session_num):
        y = []
        for ndx,i in enumerate(self.y[session_num]):
            if ':' in i: 
                y.append(i)
                end_ndx = ndx

        plt.xlabel('Time')
        plt.ylabel('Rate')
        plt.title('Sleep Session %d' % session_num)
        plt.xticks(np.arange(0, len(y), 5))
        plt.plot(y, np.array(self.x[session_num][15:end_ndx+1]).astype(float))
        plt.show()

if __name__ == "__main__":

    sleep_session = Sleep(SLEEP_EXPORT_PATH)
    sleep_session.plot_sleep_session(8)