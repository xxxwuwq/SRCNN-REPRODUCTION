import numpy as np
import cv2 as cv
import os
import h5py
from PIL import Image

np.set_printoptions(threshold=np.inf)

# h5文件读取
def read_h5_file(path):
    with h5py.File(path, 'r') as hf:
        hf_data = hf.get('input')
        data = np.array(hf_data)
        hf_label = hf.get('label')
        label = np.array(hf_label)
        return data, label

# 数据迭代器
def data_iterator(data, label, batch_size):
    num_examples = data.shape[0]
    num_batch = num_examples // batch_size
    num_total = num_batch * batch_size
    while True:
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        shuf_data = data[perm]
        shuf_label = label[perm]
        for i in range(0, num_total, batch_size):
            batch_data = shuf_data[i:i+batch_size]
            batch_label = shuf_label[i:i+batch_size]
            yield batch_data, batch_label

if __name__ == '__main__':
    data, label = read_h5_file('../datasets/training/training_data_91_image_patches.h5')
    print(data.shape)
    print(label.shape)



