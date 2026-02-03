import os
import cv2
import numpy as np
import nibabel as nib
import tensorflow as tf

VOLUME_SLICES = 75
VOLUME_START_AT = 0
IMG_SIZE = 192
TRAIN_DATASET_PATH = './BraTS2021_Training_Data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.SRC_PATH = TRAIN_DATASET_PATH
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, Z, W, y = self.__data_generation(Batch_ids)
        return [X, Z, W], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 2))
        Z = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 1))
        W = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 1))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(self.SRC_PATH, i)
            seg = nib.load(os.path.join(case_path, f'{i}-seg.nii.gz')).get_fdata()
            flair = nib.load(os.path.join(case_path, f'{i}-t2f.nii.gz')).get_fdata()
            ce = nib.load(os.path.join(case_path, f'{i}-t1c.nii.gz')).get_fdata()
            t1 = nib.load(os.path.join(case_path, f'{i}-t1n.nii.gz')).get_fdata()
            t2 = nib.load(os.path.join(case_path, f'{i}-t2w.nii.gz')).get_fdata()

            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                X[j + VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                W[j + VOLUME_SLICES*c,:,:,0] = cv2.resize(t1[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                Z[j + VOLUME_SLICES*c,:,:,0] = cv2.resize(t2[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                y[j + VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT]

        y[y==4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
        return (X - X.mean()) / X.std(), (Z - Z.mean()) / Z.std(), (W - W.mean()) / W.std(), Y