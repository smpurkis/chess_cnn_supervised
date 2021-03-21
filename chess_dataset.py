import numpy as np
import tensorflow as tf


class ChessDataset(tf.keras.utils.Sequence):
    def __init__(self, batch_size, validation=False, validation_split=0.1, random_seed=1, *args, **kwargs):
        self.batch_size = batch_size
        self.tensors_filename = "data/tensors"
        self.metas_filename = "data/metas"
        self.targets_filename = "data/targets"
        self.tensors = np.memmap(self.tensors_filename, mode="r", dtype=np.int16).reshape((-1, 8, 8, 8))
        self.metas = np.memmap(self.metas_filename, mode="r", dtype=np.int16).reshape((-1, 8))
        self.targets = np.memmap(self.targets_filename, mode="r", dtype=np.int16)

        self.length = int(self.tensors.shape[0] * (validation_split if validation else (1 - validation_split)))
        self.tensors = self.tensors[:self.length]
        self.metas = self.metas[:self.length]
        self.targets = self.targets[:self.length]

    def __len__(self):
        return int(self.length / self.batch_size)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.length)
        tensors = tf.convert_to_tensor(self.tensors[start:end], dtype=tf.int32)
        metas = tf.convert_to_tensor(self.metas[start:end], dtype=tf.int32)
        targets = self.targets[start:end]
        targets = tf.one_hot(targets, depth=64*64)
        return [tensors, metas], targets
