import pickle
import os
import urllib.request
import tarfile
import numpy as np
from check import seed_init

def load_labels(file='./dataset/cifar10/batches.meta'):
    with open(file, 'rb') as f:
        dict = pickle.load(f,encoding='bytes')
    return dict

def load_batch(batch_file):
    print(batch_file)
    with open(batch_file,'rb') as f:
        dict = pickle.load(f,encoding='bytes')
    return dict

# Input: data (1*3072 matrix)
def get_image(data):
    R = data[0:1024]
    G = data[1024:2048]
    B = data[2048:3072]

    R = R.reshape(32, 32)
    G = G.reshape(32, 32)
    B = B.reshape(32, 32)

    data_reshaped = np.stack([R, G, B], axis=2)
    return data_reshaped


def normalization(data):
    return data / 255.0

def load_cifar_10(file_path='./dataset/cifar10/', val_ratio=0.1, seed=0):
    '''
    returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    '''
    print('Loading cifar10')
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    batch_num = 5
    for i in range(batch_num):
        file_name = f"{file_path}data_batch_{i+1}"
        batch_label_data = load_batch(file_name)
        X_train.append(batch_label_data[b'data'])
        Y_train.append(batch_label_data[b'labels'])

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    shuffle_indices = np.arange(X_train.shape[0])
    seed_init(seed)
    np.random.shuffle(shuffle_indices)
    
    val_size = int(val_ratio * X_train.shape[0])
    val_indices = shuffle_indices[:val_size]
    train_indices = shuffle_indices[val_size:]
    
    X_val, Y_val = X_train[val_indices], Y_train[val_indices]
    X_train, Y_train = X_train[train_indices], Y_train[train_indices]

    test_batch = load_batch(f'{file_path}test_batch')
    X_test = test_batch[b'data']
    Y_test = test_batch[b'labels']
    return normalization(X_train), Y_train, normalization(X_val), Y_val, normalization(X_test), Y_test

from tqdm import tqdm
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_cifar10(url, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    tar_filename = os.path.join(dataset_dir, 'cifar-10-python.tar.gz')
    
    # Download dataset if it does not exist
    if not os.path.exists(tar_filename):
        print("Downloading CIFAR-10 dataset...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=tar_filename, reporthook=t.update_to, data=None)
        print("Download complete.")

    # Extract dataset
    with tarfile.open(tar_filename, 'r:gz') as tar:
        print("Extracting files...")
        for member in tar.getmembers():
            # Only extract files and modify the path to avoid the cifar-10-batches-py folder
            if member.isfile() and member.name.startswith('cifar-10-batches-py/'):
                member.name = os.path.basename(member.name)  # Change the member name to only the filename
                tar.extract(member, path=dataset_dir)  # Extract it to the dataset_dir directly
        print("Extraction complete.")


