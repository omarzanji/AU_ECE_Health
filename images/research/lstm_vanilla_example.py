import numpy as np
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

SEQ = 30

def generate_data(step_size=SEQ, y_vals=[]):
    if len(y_vals) == 0:
        x_line = np.arange(-SEQ*np.pi, SEQ*np.pi, 0.1) # x
        data = np.sin(x_line) # y
    else:
        data = y_vals
    step_size = SEQ
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
        model.add(LSTM(SEQ, input_shape=(SEQ, 1)))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
        return model

    def train_model(self, model):
        hist = model.fit(self.x, self.y, epochs=4, verbose=1)
        model.summary()
        model.save('models/actigraphy.model')
        return hist

    def generate(self, model, vals=[]):
        if len(vals)==0:
            test_x = np.arange(0, SEQ*np.pi, 0.1)
            func = np.sin(test_x)
            test_y = func[:SEQ]
        else:
            test_x = vals[0]
            func = vals[1]
            test_y = vals[1][:SEQ]

        result = []

        for x in range(len(test_x) - SEQ):
            inp = test_y[x : x+SEQ]
            inp = inp.reshape((1, SEQ, 1))
            y = model.predict(inp, verbose=0)
            test_y = np.append(test_y, y)

        x = np.arange(0,len(test_x))

        fig = plt.figure(0)
        plt.plot(x[:SEQ], test_y[:SEQ], label='input', color='red')
        plt.plot(x[SEQ:], test_y[SEQ:], label='pred', color='blue')
        fig = plt.figure(1)
        plt.plot(x, func, label='actual', color='green')
        plt.legend()

def load_sleep_data():
    with open('data/sleep_as_android_data.json', 'r') as f:
        data = json.load(f)
    return np.array(data['x']), np.array(data['y'])


def load_actigraphy_data():
    x = []
    y = []
    num = 0
    with open('data/actigraphy.json', 'r') as f:
        data = json.load(f)
        keys = data.keys()
        for subject in keys:
            subject_data = data[subject]
            num+=1
            if num == 5: break
            for sample in subject_data:
                x.append(sample[3])
                y.append(sample[4])
        x_t = []
        y_t = []
        for test in data['5030']:
            x_t.append(test[3])
            y_t.append(test[4])

    return np.array(x), np.array(y), np.array(x_t), np.array(y_t)

if __name__=="__main__":
    # x_train, y_train = generate_data(SEQ)
    # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    # sin_gen = LSTM_model(x_train, y_train)
    # sin_model = sin_gen.create_model()
    # history = sin_gen.train_model(sin_model)
    # plt.plot(history.history['loss'], label="loss")
    # plt.legend(loc="upper right")
    # sin_gen.generate(sin_model)
    # plt.show()
    # exit()

    # x, y = load_sleep_data()
    # x_t = x[0]
    # y_t = y[0]
    # # x = np.delete(x,4,0)
    # # y = np.delete(y,4,0)
    # data = []
    # for vals in y:
    #     for val in vals:
    #         data.append(val)
    # x_train, y_train = generate_data(SEQ, data)
    # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # sleep_gen = LSTM_model(x_train, y_train)
    # sleep_model = sleep_gen.create_model()
    # hist = sleep_gen.train_model(sleep_model)

    # fig = plt.figure(2)
    # plt.plot(hist.history['loss'], label="loss")
    # plt.legend(loc="upper right")
    # sleep_gen.generate(sleep_model,(np.array(x_t),np.array(y_t)))
    # plt.show()
    # exit()

    x, y, x_test, y_test = load_actigraphy_data()
    x_plot = np.arange(len(x_test[0:700]))
    y_plot = y_test[0:700]
    plt.plot(x_plot, y_plot)
    plt.show()
    x_train, y_train = generate_data(SEQ, y)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    sleep_gen = LSTM_model(x_train, y_train)
    sleep_model = sleep_gen.create_model()
    hist = sleep_gen.train_model(sleep_model)

    fig = plt.figure(2)
    plt.plot(hist.history['loss'], label="loss")
    plt.legend(loc="upper right")
    sleep_gen.generate(sleep_model,(np.array(x[0:1000]),np.array(y[0:1000])))
    plt.show()
    exit()