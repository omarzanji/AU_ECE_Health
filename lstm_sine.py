import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation

import matplotlib.pyplot as plt

def generate_data(step_size=20):
    x_line = np.arange(-50*np.pi, 50*np.pi, 0.1)
    data = np.sin(x_line)

    step_size = 20
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
        model.add(LSTM(20, input_shape=(20, 1)))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, model):
        hist = model.fit(self.x, self.y, epochs=4, verbose=1)
        model.summary()
        return hist

    def generate(self, model):
        test_x = np.arange(0, 50*np.pi, 0.1)
        func = np.cos(test_x)

        test_y = func[:20]
        
        result = []

        for x in range(len(test_x) - 20):
            inp = test_y[x : x+20]
            inp = inp.reshape((1, 20, 1))
            y = model.predict(inp, verbose=0)
            test_y = np.append(test_y, y)

        plt.plot(test_y[20:], label='pred', color='blue')
        plt.plot(test_y[:20], label='input', color='red')
        plt.legend()
        plt.show()



if __name__=="__main__":
    x_train, y_train = generate_data(20)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    sin_gen = LSTM_model(x_train, y_train)
    sin_model = sin_gen.create_model()
    history = sin_gen.train_model(sin_model)
    # plt.plot(history.history['loss'], label="loss")
    # plt.legend(loc="upper right")

    sin_gen.generate(sin_model)