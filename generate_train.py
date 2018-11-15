import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import h5py
import math

def search(dirname):
    train_name = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filenname = os.path.join(dirname, filenames)
        ext = os.path.splitext(full_filenname)[-1]
        if ext == ',bmp':
            train_name.append(full_filenname)
    return train_name

def modcrop(imgs, modulo):
    if np.size(imgs.shape) == 3:
        (sheight, swidth, _) = imgs.shape
        sheight = sheight - np.mod(sheight, modulo)
        swidth = swidth - np.mod(swidth, modulo)
        imgs = imgs[0:sheight, 0:swidth, :]
    else:
        (sheight, swidth) = imgs.shape
        sheight = sheight - np.mod(sheight, modulo)
        swidth = swidth - np.mod(swidth, modulo)
        imgs = imgs[0:sheight, 0:swidth]

    return imgs

folder = r'../datasets/images/train'
savepath = '../datasets/training/training_data_91_image_patches.h5'
size_input = 33
size_label = 21
scale = 3
stride = 14
count = 0
padding = (size_input - size_label) // 2
data = []
label = []
input_images = []
label_images = []

for(root, dir, files) in os.walk(folder):
    for file in files:
        filepath = root + '/' + file
        image = cv.imread(filepath)
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        image = image[:, :, 0:3]
        im_label = modcrop(image, scale)
        (hei, wid, _) = im_label.shape
        im_input = cv.resize(im_label, (0, 0), fx=1.0/scale, fy=1.0/scale, interpolation=cv.INTER_CUBIC)
        im_input = cv.resize(im_input, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

        # low resolution for input
        im_input = im_input.astype('float32') / 255.0
        # high resolution for label
        im_label = im_label.astype('float32') / 255.0

        input_images.append(im_input)
        label_images.append(im_label)
        for x in range(0, hei - size_input + 1, stride):
            for y in range(0, wid - size_input + 1, stride):
                sub_im_input = im_input[x:x + size_input, y:y + size_input, 0]
                sub_im_label = im_label[x + padding:x + padding + size_label, y + padding: y + padding + size_label, 0]
                sub_im_input = sub_im_input.reshape([size_input, size_input, 1])
                sub_im_label = sub_im_label.reshape([size_label, size_label, 1])

                data.append(sub_im_input)
                label.append(sub_im_label)
                count = count + 1
data = np.asarray(data)
label = np.asarray(label)
print(data.shape)
with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('input', data=data)
    hf.create_dataset('label', data=label)

