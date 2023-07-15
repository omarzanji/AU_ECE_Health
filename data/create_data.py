from cProfile import label
import pickle
import pandas as pd
import wget
import h5py
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import signal


def get_dodh_data():
    urls = [
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/3e842aa8-bcd9-521e-93a2-72124233fe2c.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/095d6e40-5f19-55b6-a0ec-6e0ad3793da0.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/0d79f4b1-e74f-5e87-8e42-f9dd7112ada5.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/119f9726-eb4c-5a0e-a7bb-9e15256149a1.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/14c012bd-65b0-56f5-bc74-2dffcea69837.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/18ede714-aba3-5ad8-bb1a-18fc9b1c4192.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/1da3544e-dc5c-5795-adc3-f5068959211f.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/1fa6c401-d819-50f5-8146-a0bb9e2b2516.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/25a6b2b0-4d09-561b-82c6-f09bb271d3be.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/37d0da97-9ae8-5413-b889-4e843ff35488.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/3e842aa8-bcd9-521e-93a2-72124233fe2c.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/5bf0f969-304c-581e-949c-50c108f62846.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/64959ac4-53b5-5868-a845-c7476e9fdf7b.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/67fa8e29-6f4d-530e-9422-bbc3aca86ed0.h5',
        'https://dreem-dod-h.s3-eu-west-3.amazonaws.com/769df255-2284-50b3-8917-2155c759fbbd.h5'

    ]
    for ndx, url in enumerate(urls):
        print(f'\ndownloading file {ndx+1} of {len(urls)}')
        try:
            wget.download(url)
        except BaseException as e:
            print(e)


def gen_dodh_data(no_spect=False):

    print('processing dodh data')

    def create_data(label_size, raw_data):
        raw_data_size = raw_data.size

        # divide number of samples with label size to see how many samples per label
        chunk_size = int(raw_data_size / label_size)

        # evenly distribute labels over signal
        chunked_data = []
        for i in range(0, raw_data_size, chunk_size):
            chunk = raw_data[i:i+chunk_size]
            if no_spect:
                chunked_data.append(chunk)
            else:
                f, t, Zxx = signal.stft(chunk)
                # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0,
                #                vmax=0.2, shading='gouraud')
                # plt.title(
                #     'PSG Spectrogram, Session 1, Epoch 1, 7500 Time-steps')
                # plt.show()
                spectrogram = np.abs(Zxx)
                chunked_data.append(spectrogram)
        return chunked_data

    sleep_sessions = dict()

    print('transforming data...')
    for ndx, filename in enumerate(os.listdir('dodh_data')):
        if ndx < len(list(os.listdir('dodh_data')))-17:
            if filename.endswith(".h5"):
                print(f'processing session {ndx+1}')
                filename = 'dodh_data/' + filename
                f = h5py.File(filename, "r")
                session = f'session {ndx+1}'
                labels = f['hypnogram'][()]
                sleep_sessions[session] = dict()
                sleep_sessions[session]['labels'] = labels

                label_size = labels.size
                eegs = f['signals']['eeg']
                emgs = f['signals']['emg']
                eogs = f['signals']['eog']

                for eeg in eegs.keys():
                    eeg_name = eeg
                    eeg_data = eegs[eeg][()]
                    data = create_data(label_size, eeg_data)
                    sleep_sessions[session][eeg_name] = data

                for emg in emgs.keys():
                    emg_name = emg
                    emg_data = emgs[emg][()]
                    data = create_data(label_size, emg_data)
                    sleep_sessions[session][emg_name] = data

                for eog in eogs.keys():
                    eog_name = eog
                    eog_data = eogs[eog][()]
                    data = create_data(label_size, eog_data)
                    sleep_sessions[session][eog_name] = data

    print('saving data...')

    if no_spect:

        with open(f'dodh_sessions_no_spect.pkl', 'wb') as f:
            pickle.dump(sleep_sessions, f)

        print(f'data saved as dodh_sessions_no_spect.pkl')

    else:

        with open(f'dodh_sessions.pkl', 'wb') as f:
            pickle.dump(sleep_sessions, f)

        print(f'data saved as dodh_sessions.pkl')

    return sleep_sessions
    # np.save('dodh_x.npy', data)
    # np.save('dodh_y.npy', labels)


def gen_urban_india():
    """
    Generates Urban Poor in India dataset for SleepNet.
    """
    SAMPLES = 17
    print(f'gathering and transforming {SAMPLES} samples...')
    df = pd.read_stata('urban_poor_data/actigraph_epochs_cleaned.dta')
    count = 0
    data = dict()
    for x in df.iterrows():
        sample = x[1]
        pid = sample[0]
        time_hr = sample[1]
        sleep_time = sample[2]
        axis1 = sample[3]
        axis2 = sample[4]
        axis3 = sample[5]
        vm = sample[6]
        lux = sample[8]
        sleep_status = sample[14]
        axis_avg = (axis1+axis2+axis3)/3
        day = sample[20]
        if pid in data:
            data[pid].append([pid, day, time_hr, sleep_time,
                             lux, sleep_status, axis1, axis2, axis3])
        else:
            count += 1
            if count == SAMPLES+1:
                break
            data[pid] = [[pid, day, time_hr, sleep_time,
                          lux, sleep_status, axis1, axis2, axis3]]
    data_keys = data.keys()
    train_data = dict()
    val_data = dict()
    for pid_count, key in enumerate(data_keys, 1):
        if pid_count <= SAMPLES-2:
            train_data[key] = data[key]
        if pid_count > SAMPLES-2:  # take the last two samples for validation
            val_data[key] = data[key]
    print(f'{len(train_data)} Training Samples, {len(val_data)} Validation Samples')
    with open('actigraphy.json', 'w') as f:
        json.dump(train_data, f)
    with open('validation_actigraphy.json', 'w') as f:
        json.dump(val_data, f)


if __name__ == "__main__":
    data = gen_dodh_data(no_spect=True)
    # get_dodh_data()
    # gen_urban_india()
