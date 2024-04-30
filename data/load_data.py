import os
import shutil  # https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive  # to unzip
# from shutil import make_archive # to create zip for storage
import requests  # for downloading zip file
from scipy import io  # for loadmat, matlab conversion
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

# credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
# many other methods I tried failed to download the file properly


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def unimib_load_dataset(
        verbose=True,
        incl_xyz_accel=False,  # include component accel_x/y/z in ____X data
        # add rms value (total accel) of accel_x/y/z in ____X data
        incl_rms_accel=True,
        incl_val_group=False,  # True => returns x/y_test, x/y_validation, x/y_train
    # False => combine test & validation groups
        split_subj=dict
    (train_subj=[4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 19, 20, 21, 22, 24, 26, 27, 29],
     validation_subj=[1, 9, 16, 23, 25, 28],
     test_subj=[2, 3, 13, 17, 18, 30]),
        one_hot_encode=True):

    # Download and unzip original dataset
    if (not os.path.isfile('./UniMiB-SHAR.zip')):
        print("Downloading UniMiB-SHAR.zip file")
        # invoking the shell command fails when exported to .py file
        # redirect link https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
        #!wget https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
        download_url(
            'https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip', './UniMiB-SHAR.zip')
    if (not os.path.isdir('./UniMiB-SHAR')):
        shutil.unpack_archive('./UniMiB-SHAR.zip', '.', 'zip')
    # Convert .mat files to numpy ndarrays
    path_in = './UniMiB-SHAR/data'
    # loadmat loads matlab files as dictionary, keys: header, version, globals, data
    adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
    adl_names = io.loadmat(path_in + '/adl_names.mat',
                           chars_as_strings=True)['adl_names']
    adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']

    # Reshape data and compute total (rms) acceleration
    num_samples = 151
    # UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
    adl_data = np.reshape(adl_data, (-1, num_samples, 3),
                          order='F')  # uses Fortran order
    if (incl_rms_accel):
        rms_accel = np.sqrt(
            (adl_data[:, :, 0]**2) + (adl_data[:, :, 1]**2) + (adl_data[:, :, 2]**2))
        adl_data = np.dstack((adl_data, rms_accel))

    # remove component accel if needed
    if (not incl_xyz_accel):
        adl_data = np.delete(adl_data, [0, 1, 2], 2)

    # matlab source was 1 indexed, change to 0 indexed
    act_num = (adl_labels[:, 0])-1
    sub_num = (adl_labels[:, 1])  # subject numbers are in column 1 of labels

    if (not incl_val_group):
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj'] +
                                         split_subj['validation_subj']))
        x_train = adl_data[train_index]
        y_train = act_num[train_index]
    else:
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj']))
        x_train = adl_data[train_index]
        y_train = act_num[train_index]

        validation_index = np.nonzero(
            np.isin(sub_num, split_subj['validation_subj']))
        x_validation = adl_data[validation_index]
        y_validation = act_num[validation_index]

    test_index = np.nonzero(np.isin(sub_num, split_subj['test_subj']))
    x_test = adl_data[test_index]
    y_test = act_num[test_index]

    if (verbose):
        print("x/y_train shape ", x_train.shape, y_train.shape)
    if (incl_val_group):
        print("x/y_validation shape ", x_validation.shape, y_validation.shape)
    print("x/y_test shape  ", x_test.shape, y_test.shape)

    if (incl_val_group):
        return x_train, y_train, x_validation, y_validation, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test
    
X_train, y_train, X_valid, y_valid, X_test, y_test = unimib_load_dataset(
    incl_xyz_accel=True, incl_val_group=True, one_hot_encode=False, verbose=True
)

# Number of classes
n_classes = len(np.unique(y_train))
print(f"Number of classes: {n_classes}")

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.int64)

X_valid = torch.tensor(X_valid, dtype=torch.float32)
Y_valid = torch.tensor(y_valid, dtype=torch.int64)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(y_test, dtype=torch.int64)

# DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

valid_dataset = TensorDataset(X_valid, Y_valid)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)