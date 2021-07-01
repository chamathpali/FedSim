# For more information refer - https://github.com/chamathpali/Fed-MEx
import os
import csv
import datetime as dt
import numpy as np
import random
import json

random.seed(0)
np.random.seed(1)

frame_size = 16*16

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

path = 'pm/'
results_file = 'pm_json.json'

frames_per_second = 1
window = 5
increment = 2

pm_min_length = frames_per_second*window
pm_max_length = 15*window


def _read(_file):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row[0]) == 19 and '.' not in row[0]:
            row[0] = row[0]+'.000000'
        temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
        _temp = [float(f) for f in row[1:]]
        temp.extend(_temp)
        _data.append(temp)
    return _data


def read():
    alldata = {}
    subjects = os.listdir(path)
    for subject in [f for f in subjects if f!='.DS_Store']:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            sensor = activity.split('.')[0].replace('_pm', '')
            activity_id = sensor.split('_')[0]
            _data = _read(os.path.join(subject_path, activity), )
            if activity_id in allactivities:
                allactivities[activity_id][sensor] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor] = _data
        alldata[subject] = allactivities
    return alldata


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def trim(_data):
    _length = len(_data)
    _inc = int(_length/(window*frames_per_second))
    _new_data = []
    for i in range(window*frames_per_second):
        _new_data.append(_data[i*_inc])
    return _new_data


def frame_reduce(_data):
    if frames_per_second == 0:
        return _data
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                time_windows.append(trim(item))
            _activities[activity] = time_windows
        _features[subject] = _activities
    return _features


def split_windows(data):
    outputs = []
    start = data[0][0]
    end = data[len(data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    frames = [a[1:] for a in data[:]]
    frames = np.array(frames)
    _length = frames.shape[0]
    frames = np.reshape(frames, (_length*frame_size))
    frames = frames/max(frames)
    frames = [float("{0:.5f}".format(f)) for f in frames.tolist()]
    frames = np.reshape(np.array(frames), (_length, frame_size))

    while start + _window < end:
        _end = start + _window
        start_index = find_index(data, start)
        end_index = find_index(data, _end)
        instances = [a[:] for a in frames[start_index:end_index]]
        start = start + _increment
        outputs.append(instances)
    return outputs


# single sensor
def extract_features(_data):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            time_windows = []
            activity_id = activity_id_dict.get(activity)
            activity_data = activities[activity]
            for sensor in activity_data:
                time_windows.extend(split_windows(activity_data[sensor]))
            _activities[activity_id] = time_windows
        _features[subject] = _activities
    return _features


def split(_data, _labels, test_indices):
    _train_data = []
    _train_labels = []
    _test_data = []
    _test_labels = []
    index = 0
    for _datum, _label in zip(_data, _labels):
        if index in test_indices:
            _test_data.append(_datum)
            _test_labels.append(_label)
        else:
            _train_data.append(_datum)
            _train_labels.append(_label)
        index += 1
    return _train_data, _train_labels, _test_data, _test_labels



def flatten(_data):
    flatten_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return flatten_data, flatten_labels


def pad(data, length):
    pad_length = []
    if length % 2 == 0:
        pad_length = [int(length / 2), int(length / 2)]
    else:
        pad_length = [int(length / 2) + 1, int(length / 2)]
    new_data = []
    for index in range(pad_length[0]):
        new_data.append(data[0])
    new_data.extend(data)
    for index in range(pad_length[1]):
        new_data.append(data[len(data) - 1])
    return new_data


def reduce(data, length):
    red_length = []
    if length % 2 == 0:
        red_length = [int(length / 2), int(length / 2)]
    else:
        red_length = [int(length / 2) + 1, int(length / 2)]
    new_data = data[red_length[0]:len(data) - red_length[1]]
    return new_data


def pad_features(_features):
    new_features = {}
    for subject in _features:
        new_activities = {}
        activities = _features[subject]
        for act in activities:
            items = activities[act]
            new_items = []
            for item in items:
                _len = len(item)
                if _len < pm_min_length:
                    continue
                elif _len > pm_max_length:
                    item = reduce(item, _len - pm_max_length)
                    new_items.append(item)
                elif _len < pm_max_length:
                    item = pad(item, pm_max_length - _len)
                    new_items.append(item)
                elif _len == pm_max_length:
                    new_items.append(item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


all_data = read()
all_features = extract_features(all_data)
all_features = pad_features(all_features)
all_features = frame_reduce(all_features)

users = list(all_features.keys())
print(users)

user_data_train_dict = {}
user_data_test_dict = {}

number_of_train_samples = []
number_of_test_samples = []

names= [ "f_"+f for f in users]
for user in users:
    user_data = all_features[user]
    all_user_data = []
    all_user_labels = []

    user_classes = np.random.choice(list(user_data.keys()), 2)
    for key in user_classes:
        all_user_data.extend(user_data[key])
        all_user_labels.extend([key for f in range(len(user_data[key]))])

    perm = np.random.permutation(len(all_user_data))
    all_user_data = np.array(all_user_data)
    all_user_labels = np.array(all_user_labels)

    rand_data =  random.randint( int(len(all_user_data)/10),len(all_user_data))
    print(rand_data)
    print(all_user_data.shape)
    print(all_user_labels.shape)
    all_user_data = np.reshape(all_user_data, (all_user_data.shape[0],
                                               all_user_data.shape[1]*
                                               all_user_data.shape[2]))

    all_user_data = all_user_data[perm][:rand_data]
    all_user_labels = all_user_labels[perm][:rand_data]
    print(all_user_data.shape)
    print(all_user_labels.shape)
    train_test_split = int(len(all_user_data)*0.8)
    all_user_data_train = all_user_data[:train_test_split]
    all_user_data_test = all_user_data[train_test_split:]
    all_user_labels_train = all_user_labels[:train_test_split]
    all_user_labels_test = all_user_labels[train_test_split:]

    print(all_user_data_train.shape)
    print(all_user_data_test.shape)
    print(all_user_labels_train.shape)
    print(all_user_labels_test.shape)

    train_data = {}
    train_data["x"] = all_user_data_train.tolist()
    train_data["y"] = all_user_labels_train.tolist()

    test_data = {}
    test_data["x"] = all_user_data_test.tolist()
    test_data["y"] = all_user_labels_test.tolist()

    user_data_train_dict["f_"+user] = train_data
    user_data_test_dict["f_"+user] = test_data

    number_of_train_samples.append(all_user_data_train.shape[0])
    number_of_test_samples.append(all_user_data_test.shape[0])

train_dict = {"users": names, "user_data": user_data_train_dict, "num_samples": number_of_train_samples}
test_dict = {"users": names, "user_data": user_data_test_dict, "num_samples": number_of_test_samples}

with open('train.json', 'w') as json_file:
    json.dump(train_dict, json_file)
with open('test.json', 'w') as json_file:
    json.dump(test_dict, json_file)


