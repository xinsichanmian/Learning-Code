import mne
import numpy as np
from datetime import datetime
import os
# import keras.utils as ku

from moabb.datasets import gigadb
from sklearn.model_selection import StratifiedKFold


def get_Epoch(raw, interval=[-2.0, 5.0], stim_channel="Stim", event_id={"left_hand": 1, "right_hand": 2}):
    """
    funs:从raw里得到可监督学习的X，Y
    """
    events = mne.find_events(raw, stim_channel=stim_channel)

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')

    epochs = mne.Epochs(raw, events, event_id=event_id, picks=picks,
                        tmin=interval[0], tmax=interval[1], baseline=None, proj=False, preload=True)

#     epochs.plot()
#     fig = mne.viz.plot_events(events, event_id=event_id, sfreq=raw.info['sfreq'],
#                               first_samp=raw.first_samp)
    return epochs


def get_Xy_fromRaw(raw, interval=[-2.0, 5.0], stim_channel="Stim", event_id={"left hand": 1, "right hand": 2}):
    """
    funs:从raw里得到可监督学习的X，Y
    MI Interval：[0., 3.]
    All Interval：[-2., 5.]

    """
    epochs = get_Epoch(raw, interval, stim_channel, event_id)
    X = epochs.get_data()

    y = epochs.events[:, -1]
#     y = ku.to_categorical(y-1)
    return X, y


def sliding_window(data, labels, window_sz, n_hop, n_start=0, show_status=False):
    """

    input: (array) data : matrix to be processed

           (int)   window_sz : nb of samples to be used in the window

           (int)   n_hop : size of jump between windows           

    output:(array) new_data : output matrix of size (None, window_sz, feature_dim)



    """

    flag = 0

    for sample in range(data.shape[0]):

        tmp = np.array([data[sample, i:i + window_sz, :]
                        for i in np.arange(n_start, data.shape[1] - window_sz, n_hop)])

        tmp_lab = np.array([labels[sample] for i in np.arange(
            n_start, data.shape[1] - window_sz, n_hop)])

        if sample % 100 == 0 and show_status == True:

            print("Sample " + str(sample) + "processed!\n")

        if flag == 0:

            new_data = tmp

            new_lab = tmp_lab

            flag = 1

        else:

            new_data = np.concatenate((new_data, tmp))

            new_lab = np.concatenate((new_lab, tmp_lab))

    return new_data, new_lab


def augment_train_data(X, y, FS=160, n_splits=5, ts=3):
    """
    n_splits : 只对训练集进行数据增强
    """
#     print(X.shape)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        x_augmented, y_augmented = sliding_window(data=np.rollaxis(X_train, 2, 1),
                                                  labels=y_train,
                                                  window_sz=ts * FS,
                                                  n_hop=FS // 10,
                                                  n_start=0)
        
        x_augmented = np.rollaxis(x_augmented, 2, 1)
        X_test = X_test[:, :, FS:FS*(ts+1)]
        validation_data = [X_test, y_test]
#         print("x_augmented.shape", x_augmented.shape)
#         print("X_train", X_train.shape)
        return x_augmented, y_augmented, validation_data
