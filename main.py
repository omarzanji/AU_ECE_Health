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
import ast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras

import xgboost as xgb

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class SleepNet:
    """
    Creates and trains LSTM sleep / wake prediction models and plots results. 
    Model is saved to models/ as a Tensorflow / Keras .model file. Data 
    is cached with domain=0 (training data) and domain=1 (validation data). 
    """

    def __init__(self, domain='UrbanPoorIndia', model_arch='LSTM', seq=10):
        self.seq = seq
        self.type = domain
        self.model_arch = model_arch
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
        

    def create_model(self, units=None):
        """
        Create LSTM model with relu activation and MSE loss.
        """
        model_arch = self.model_arch
        model = Sequential()
        if self.type == 'UrbanPoorIndia':
            xshape = self.x.shape[1]
            yshape = 1
            print(f'xshape: {xshape}', f'yshape: {yshape}')
            if model_arch == 'LSTM':
                model.add(LSTM(units, input_shape=(xshape,self.seq)))
                model.add(Activation('relu'))
                model.add(Dense(yshape))
                model.add(Activation('relu'))
                model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics='accuracy')
                return model
            elif model_arch == 'XGBoost': # XGBoost goes here...
                model = xgb.XGBRegressor(
                    max_depth=7,
                    learning_rate=0.1
                )
                return model
        else:
            xshape = self.x.shape[1]
            yshape = self.y.shape[1]
            print(f'xshape: {xshape}', f'yshape: {yshape}')
            if model_arch == 'LSTM':
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
            else: # other model architectures go here...
                pass


    def train_model(self, model=None, name=None, epochs=None, learning_rate=None, rounds=None, depth=None):
        """
        Train model with optimal parameters.
        """
        print(f'\n[Training {self.model_arch} Model...]\n\n')
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y)
        if self.model_arch == 'LSTM':
            self.hist = model.fit(xtrain, ytrain, epochs=epochs, verbose=1)
            self.plot_history(self.hist)
            model.summary()
            self.ypreds = model.predict(xtest)
            self.get_accuracy(ytest, self.ypreds)
            self.ytest = ytest
            model.save(f'models/{name}')
        elif self.model_arch == 'XGBoost':
            self.xtrain = xtrain
            self.ytrain = ytrain
            self.ytest = ytest
            xtrain = []
            for x in self.xtrain:
                xtrain.append(x.flatten())
            # model.fit(xtrain, ytrain)
            dtrain = xgb.DMatrix(data=xtrain, label=ytrain)
            params = {
                'max_depth':depth,
                'learning_rate':learning_rate,
                'objective': 'reg:squarederror',
                'eval_metric':'logloss'
            }
            rounds = rounds
            evals_result = {}
            
            watchlist = [(dtrain, 'eval')]
            model = xgb.train(params, dtrain, rounds, watchlist, evals_result=evals_result)

            xtest_ = []
            for x in xtest:
                xtest_.append(x.flatten())
            dtest = xgb.DMatrix(data=xtest_)
            self.ypreds = model.predict(dtest)
            accuracy = self.get_accuracy(ytest, self.ypreds)
            self.results = evals_result
            results = self.results
            return model, results, accuracy
            # plt.plot(results['eval']['logloss'], label='logloss')
            # # show the legend
            # plt.legend()
            # # show the plot
            # plt.show()


    def get_accuracy(self, yT, yP):
        if self.type == 'UrbanPoorIndia':
            accuracy = accuracy_score(yT, yP.round())
        else:
            y_truth = []
            y_pred = []
            for i in range(len(yT)):
                y_truth.append(np.argmax(yT[i]))
                y_pred.append(np.argmax(yP[i]))
            accuracy = accuracy_score(y_truth, y_pred)
        print(f'\nAccuracy: {accuracy}\n')
        return accuracy

    def train_model_sweep(self):
        """
        Test different model parameters to know how to fine-tune accuracy.
        """
        print(f'\n[Starting {self.model_arch} Parameter Sweep...]\n\n')

        if self.model_arch == 'LSTM':
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
                testing_accuracy = sweep_dict[key][1]
                self.sweep_dict[str(key)] = {"loss": hist['loss'], "accuracy": hist['accuracy'], "testing_accuracy": testing_accuracy}
            with open(f'{self.type}_{self.model_arch}_param_sweep.json', 'w') as f:
                json.dump(self.sweep_dict, f)
                
        elif self.model_arch == 'XGBoost':
            rounds_ = [5,10,20]
            learning_rates_ = [0.1, 0.5, 0.7]
            depths_ = [1,5,7]
            sweep_dict = {}
            for rounds in rounds_:
                sweep_dict[rounds] = []
                for learning_rate in learning_rates_:
                    for depth in depths_:
                        print(f'\n[Training with {rounds} rounds, {learning_rate} learning rate, and max depth {depth}]\n')
                        model, results, accuracy = self.train_model(learning_rate=learning_rate, rounds=rounds, depth=depth)
                        sweep_dict[rounds].append([str((learning_rate, depth)), results['eval']['logloss'], accuracy])
            
            with open(f'{self.type}_{self.model_arch}_param_sweep.json', 'w') as f:
                json.dump(sweep_dict, f)

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

        # generate bar plot of min, average, and max accuracy results 
        def make_acc_bar_plot(acc_x_arr, acc_y_arr):
            min = np.min(acc_y_arr)
            min_ndx = acc_y_arr.index(min)
            max = np.max(acc_y_arr)
            max_ndx = acc_y_arr.index(max)
            avg = np.average(acc_y_arr)
            for val in acc_y_arr:
                if val >= avg:
                    avg_val = val
                    break
            avg_ndx = acc_y_arr.index(avg_val)
            x = [
                acc_x_arr[min_ndx],
                acc_x_arr[avg_ndx],
                acc_x_arr[max_ndx]
            ]
            y = [
                acc_y_arr[min_ndx],
                acc_y_arr[avg_ndx],
                acc_y_arr[max_ndx]
            ]
            arr = ['min', 'avg', 'max']
            for x_,y_ in zip(x,y):
                plt.bar(x_,y_, label=f'{arr[x.index(x_)]}', width=0.3)
                plt.text(x_,y_,str(y_*100)[0:5], ha='center', va='bottom')

        with open(f'{self.type}_{self.model_arch}_param_sweep.json', 'r') as f:
            param_sweep = json.load(f)

        if self.model_arch=='LSTM':
            
            # Generate loss curves plot and collect testing accuracies for bar plot later
            plt.figure()
            test_acc_x = []
            test_acc_y = []
            for key in param_sweep.keys():
                units = ast.literal_eval(key)[0]
                epochs = ast.literal_eval(key)[1]
                loss_curve = param_sweep[key]['loss']
                testing_accuracy = param_sweep[key]['testing_accuracy']
                test_acc_x.append(f'{units} units, {epochs} epochs')
                test_acc_y.append(testing_accuracy)
                plt.plot(loss_curve, label=f'{units} units, {epochs} epochs')
            plt.ylabel('loss')
            plt.xlabel('Training epochs')
            plt.title(f'Parameter Sweep Losses on {self.type} LSTM Model')
            plt.yscale('log')
            plt.legend()

            # Generate accuracy curve plots
            plt.figure()
            for key in param_sweep.keys():
                units = ast.literal_eval(key)[0]
                epochs = ast.literal_eval(key)[1]
                plt.plot(param_sweep[key]['accuracy'], label=f'{units} units, {epochs} epochs')
            plt.ylabel('accuracy')
            plt.xlabel('Training epochs')
            plt.title(f'Parameter Sweep Accuracies on {self.type} LSTM Model')
            plt.yscale('log')
            plt.legend()

            # Make bar plot
            make_acc_bar_plot(test_acc_x, test_acc_y)

            plt.title('Accuracies from LSTM Parameter Sweep')
            plt.tight_layout()
            plt.legend()
            plt.ylim((0, 1.2))
            plt.xlabel('Parameters')
            plt.ylabel('Accuracy')

            plt.show()

        elif self.model_arch == 'XGBoost':
            
            acc_x_arr = []
            acc_y_arr = []
            plt.figure()

            # sweep_dict[rounds].append([str((learning_rate, depth)), results['eval']['logloss'], accuracy])
            for rounds in param_sweep.keys():
                rounds_sweep = param_sweep[rounds]
                for model in rounds_sweep:
                    label = ast.literal_eval(model[0])
                    learning_rate = label[0]
                    depth = label[1]
                    training_loss_curve = model[1]
                    plt.plot(training_loss_curve, label=f'learning_rate: {learning_rate}, depth: {depth}')
                    accuracy = model[2]
                    a_x = (rounds, learning_rate, depth)
                    a_y = accuracy
                    acc_x_arr.append(str(a_x))
                    acc_y_arr.append(a_y)

                plt.title('XGBoost Parameter Sweep Training Loss')
                plt.xlabel('Training rounds')
                plt.ylabel('Logloss')
                plt.legend()
                plt.figure()

            min = np.min(acc_y_arr)
            min_ndx = acc_y_arr.index(min)
            
            max = np.max(acc_y_arr)
            max_ndx = acc_y_arr.index(max)
            avg = np.average(acc_y_arr)
            for val in acc_y_arr:
                if val >= avg:
                    avg_val = val
                    break
            avg_ndx = acc_y_arr.index(avg_val)
            
            x = [
                acc_x_arr[min_ndx],
                acc_x_arr[avg_ndx],
                acc_x_arr[max_ndx]
            ]
            y = [
                acc_y_arr[min_ndx],
                acc_y_arr[avg_ndx],
                acc_y_arr[max_ndx]
            ]
            arr = ['min', 'avg', 'max']
            for x_,y_ in zip(x,y):

                plt.bar(x_,y_, label=f'{arr[x.index(x_)]}', width=0.3)
                plt.text(x_,y_,str(y_*100)[0:5], ha='center', va='bottom')
            plt.title('Accuracies from XGBoost Parameter Sweep')
            plt.tight_layout()
            plt.legend()
            plt.ylim((0, 1.2))
            plt.xlabel('(rounds, learning rate, depth)')
            plt.ylabel('Accuracy')
            plt.show()

    def get_model_metrics(self, model=None):
        print('\n[Getting model metrics...]\n')
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y)
        if model==None:
            model = keras.models.load_model(f'models/{self.type}.model')
        ypreds = model.predict(xtest).round()
        accuracy = accuracy_score(ytest, ypreds)
        f1 = f1_score(ytest, ypreds)
        print(accuracy, f1)

            
if __name__ == "__main__":
    MODE = 2 # 1 for training / 2 for param sweep training / 3 for metrics
    DOMAIN = 1
    MODEL = 0

    domains = ['SleepAsAndroid', 'UrbanPoorIndia', 'AASM']
    models = ['LSTM', 'XGBoost']

    net = domains[DOMAIN]
    arch = models[MODEL]

    if MODE == 1: # Train a new model
        sleepnet = SleepNet(net, model_arch=arch, seq=10)
        model = sleepnet.create_model(units=256)
        sleepnet.train_model(model, net+'.model', epochs=15)
    elif MODE == 2: # train multiple models with param sweep
        sleepnet = SleepNet(net, model_arch=arch)
        sleepnet.train_model_sweep()
    elif MODE == 3: # plot loss curves for param sweep training
        sleepnet = SleepNet(net, model_arch=arch)
        # sleepnet.get_model_metrics()
        sleepnet.plot_param_sweep()
