'''
Predicts sleep / awake from actigraphy data using time-series forecasting.

author: Omar Barazanji
date: 3/21/2022
organizaion: Auburn University ECE
'''

import json
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class SleepNet:
    """
    Creates and trains LSTM sleep / wake prediction models and plots results. 
    Model is saved to models/ as a Tensorflow / Keras .model file. Data 
    is cached with domain=0 (training data) and domain=1 (validation data). 
    """

    def __init__(self, domain='UrbanPoorIndia', seq=10):
        self.seq = seq
        self.type = domain

        x_cache_str = f'type{domain}_SEQ{seq}_x.npy'
        y_cache_str = f'type{domain}_SEQ{seq}_y.npy'
        self.x = []
        self.y = []
        if x_cache_str in os.listdir('cache'):
            print('\n[Found cached processed data! Loading...]')
            self.x = np.load('cache/'+x_cache_str)
            self.y = np.load('cache/'+y_cache_str)
        else:
            print(f'\n[No X and Y cache found for type: {domain}, seq: {seq}]')


    def create_model(self, units=256):
        """
        Create LSTM model with relu activation and MSE loss.
        """
        model = Sequential()
        if self.type == 'UrbanPoorIndia':
            xshape = self.x.shape[1]
            yshape = 1
            print(f'xshape: {xshape}', f'yshape: {yshape}')
            model.add(LSTM(units, input_shape=(xshape,self.seq)))
            model.add(Activation('relu'))
            model.add(Dense(yshape))
            model.add(Activation('relu'))
            model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics='accuracy')
            return model
        else:
            xshape = self.x.shape[1]
            yshape = self.y.shape[1]
            print(f'xshape: {xshape}', f'yshape: {yshape}')
            model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=(xshape,self.seq)))
            model.add(Bidirectional(LSTM(units)))
            # model.add(Activation('relu'))
            model.add(Dropout(0.2))
            # model.add(LSTM(units, input_shape=(xshape,self.seq)))
            # model.add(Activation('sigmoid'))
            model.add(Dense(yshape))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics='accuracy')
            return model


    def train_model(self, model, name, epochs=5):
        """
        Train model with optimal parameters.
        """
        print('\n[Training Model...]\n\n')
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y)
        self.hist = model.fit(xtrain, ytrain, epochs=epochs, verbose=1)
        self.plot_history(self.hist)
        model.summary()
        self.ypreds = model.predict(xtest)
        self.get_accuracy(ytest, self.ypreds)
        self.ytest = ytest
        model.save(f'models/{name}')

    def get_accuracy(self, yT, yP):
        if self.type == 'UrbanPoor':
            accuracy = accuracy_score(yT, yP.round())
        else:
            y_truth = []
            y_pred = []
            for i in range(len(yT)):
                y_truth.append(np.argmax(yT[i]))
                y_pred.append(np.argmax(yP[i]))
            accuracy = accuracy_score(y_truth, y_pred)
        print(f'\nAccuracy: {accuracy}\n')

    def train_model_sweep(self):
        """
        Test different model parameters to know how to fine-tune accuracy.
        """
        print('\n[Starting Parameter Sweep...]\n\n')

        units_ = [64, 128, 256, 512]
        # units_ = [8, 32, 64]
        epochs_ = [2, 5, 10, 15]
        # epochs_ = [2, 5, 10]
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y)

        sweep_dict = {}
        for i in range(len(units_)):
            for w in range(len(epochs_)):
                units,epochs = units_[i], epochs_[w]
                print(f'\n[Training with {units} units and {epochs} epoch(s)]\n')
                model = self.create_model(units)
                self.hist = model.fit(xtrain, ytrain, epochs=epochs, verbose=1)
                self.ypreds = model.predict(xtest)
                accuracy = accuracy_score(ytest, self.ypreds.round())
                sweep_dict[str((units,epochs))] = [self.hist, accuracy]

        keys = sweep_dict.keys()
        self.sweep_dict = {}
        for key in keys:
            hist = sweep_dict[key][0].history
            self.sweep_dict[str(key)] = {"loss": hist['loss'], "accuracy": hist['accuracy']}
        with open(f'{self.type}_param_sweep.json', 'w') as f:
            json.dump(self.sweep_dict, f)

    def plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper right')

        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.title('Model Training Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()


    def plot_param_sweep(self):
        with open(f'{self.type}_param_sweep.json', 'r') as f:
            param_sweep = json.load(f)
        plt.figure()
        for key in param_sweep.keys():
            plt.plot(param_sweep[key]['loss'], label=key+' loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title(f'Param Sweep on {self.type} LSTM Model')
        plt.yscale('log')
        plt.legend()

        plt.figure()
        for key in param_sweep.keys():
            plt.plot(param_sweep[key]['accuracy'], label=key+' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.title('Param Sweep on {self.type} LSTM Model')
        plt.yscale('log')
        plt.legend()

        plt.show()
            
if __name__ == "__main__":
    TYPE = 2 # 1 for training / 2 for param sweep training / 3 for plotting param sweep results
    
    DOMAIN = 1
    domains = ['SleepAsAndroid', 'UrbanPoorIndia', 'AASM']
    net = domains[DOMAIN]
    seq = 10

    if TYPE == 1: # Train a new model
        sleepnet = SleepNet(net, seq=seq)
        model = sleepnet.create_model(units=256)
        sleepnet.train_model(model, net+'.model', epochs=15)
    elif TYPE == 2: # train multiple models with param sweep
        sleepnet = SleepNet(net)
        sleepnet.train_model_sweep()
    elif TYPE == 3: # plot loss curves for param sweep training
        sleepnet = SleepNet(net)
        sleepnet.plot_param_sweep()
