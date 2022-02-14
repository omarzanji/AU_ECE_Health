import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def generate_data(step_size=40, y_vals=None):
    if y_vals == None:
        x_line = np.arange(-40*np.pi, 40*np.pi, 0.1) # x
        data = np.sin(x_line) # y
    else:
        data = y_vals
    step_size = 40
    x = []
    y = []
    end_ndx = 0
    for ndx,val in enumerate(data):
        seq = data[ndx : ndx+step_size]
        end_ndx = ndx+step_size
        try:
            next_point = data[end_ndx]
        except IndexError:
            x.append(seq)
            y.append(next_point)
            break
        x.append(seq)
        y.append(next_point)
    return np.array(x),np.array(y)


class LSTM_model:

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
    
    def create_model(self):
        model = Sequential()
        model.add(LSTM(40, input_shape=(40, 1)))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, model):
        hist = model.fit(self.x, self.y, epochs=4, verbose=1)
        model.summary()
        return hist

    def generate(self, model, vals=None):
        if vals==None:
            test_x = np.arange(0, 40*np.pi, 0.1)
            func = np.sin(test_x)
            test_y = func[:40]
        else:
            test_x = vals[0]
            func = vals[1]
            test_y = vals[1][:40]

        result = []

        for x in range(len(test_x) - 40):
            inp = test_y[x : x+40]
            inp = inp.reshape((1, 40, 1))
            y = model.predict(inp, verbose=0)
            test_y = np.append(test_y, y)

        x = np.arange(0,len(test_x))

        fig = plt.figure(0)
        plt.plot(x[:40], test_y[:40], label='input', color='red')
        plt.plot(x[19:], test_y[19:], label='pred', color='blue')
        # fig = plt.figure(1)
        plt.plot(x, func, label='actual', color='green')
        plt.legend()

def load_sleep_data():
    import json
    with open('data.json', 'r') as f:
        data = json.load(f)
    return np.array(data['x']), np.array(data['y'])

if __name__=="__main__":
    x_train, y_train = generate_data(40)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    sin_gen = LSTM_model(x_train, y_train)
    sin_model = sin_gen.create_model()
    history = sin_gen.train_model(sin_model)
    plt.plot(history.history['loss'], label="loss")
    plt.legend(loc="upper right")
    sin_gen.generate(sin_model)
    plt.show()
    exit()
    # x, y = load_sleep_data()
    # x_t = x[0]
    # y_t = y[0]
    # # x = np.delete(x,4,0)
    # # y = np.delete(y,4,0)
    # data = []
    # for vals in y:
    #     for val in vals:
    #         data.append(val)
    # x_train, y_train = generate_data(40, data)
    # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # sleep_gen = LSTM_model(x_train, y_train)
    # sleep_model = sleep_gen.create_model()
    # hist = sleep_gen.train_model(sleep_model)

    # fig = plt.figure(2)
    # plt.plot(hist.history['loss'], label="loss")
    # plt.legend(loc="upper right")
    # sleep_gen.generate(sleep_model,(np.array(x_t),np.array(y_t)))
    # plt.show()