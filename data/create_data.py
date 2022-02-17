import pandas as pd

SAMPLES = 30

df = pd.read_stata('actigraph_epochs_cleaned.dta')

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
        data[pid].append([pid, day, time_hr, sleep_time, lux, sleep_status, axis1, axis2, axis3])
    else:
        count+=1
        if count == 2: break
        data[pid] = [[pid, day, time_hr, sleep_time, lux, sleep_status, axis1, axis2, axis3]]


import json
with open('actigraphy.json', 'w') as f:
    json.dump(data, f)