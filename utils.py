import numpy as np
import cv2 as cv
import os
import h5py

np.set_printoptions(threshold=np.inf)


def search(dirname):
    train_name = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filenname = os.path.join(dirname, filename)
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


def generate_training_data_patches(training_root, save_path, input_size, label_size, scale_factor):
    stride = 14
    count = 0
    padding = (input_size - label_size) // 2

    data = []
    label = []

    input_images = []
    label_images = []

    for (root, dir, files) in os.walk(training_root):
        for file in files:
            filepath = root + '/' + file
            image = cv.imread(filepath)
            image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
            image = image[:, :, 0:3]
            im_label = modcrop(image, scale_factor)
            (hei, wid, _) = im_label.shape
            im_input = cv.resize(im_label, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor,
                                 interpolation=cv.INTER_CUBIC)
            im_input = cv.resize(im_input, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)

            # low resolution for input
            im_input = im_input.astype('float32') / 255.0
            # high resolution for label
            im_label = im_label.astype('float32') / 255.0

            input_images.append(im_input)
            label_images.append(im_label)
            for x in range(0, hei - input_size + 1, stride):
                for y in range(0, wid - input_size + 1, stride):
                    sub_im_input = im_input[x:x + input_size, y:y + input_size, 0]
                    sub_im_label = im_label[x + padding:x + padding + label_size, y + padding: y + padding + label_size,
                                   0]
                    sub_im_input = sub_im_input.reshape([input_size, input_size, 1])
                    sub_im_label = sub_im_label.reshape([label_size, label_size, 1])

                    data.append(sub_im_input)
                    label.append(sub_im_label)
                    count = count + 1
    data = np.asarray(data)
    label = np.asarray(label)
    print(data.shape)
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('input', data=data)
        hf.create_dataset('label', data=label)


def psnr(a, b):
    diff = np.abs(a - b)
    rmse = np.sqrt(diff).sum()
    psnr = 20 * np.log10(255 / rmse)
    return psnr

def read_h5_file(path):
    with h5py.File(path, 'r') as hf:
        hf_data = hf.get('input')
        data = np.array(hf_data)
        hf_label = hf.get('label')
        label = np.array(hf_label)
        return data, label


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
            batch_data = shuf_data[i:i + batch_size]
            batch_label = shuf_label[i:i + batch_size]
            yield batch_data, batch_label


def set_gpu(gpu=0):
    """
    the gpu used setting
    :param gpu: gpu id
    :return:
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


if __name__ == '__main__':
    # generate trianing data
    # training_root = r'./datasets/train'
    # save_path = './datasets/training_data_91_image_patches.h5'
    # input_size = 33
    # label_size = 21
    # scale_factor = 3
    # generate_training_data_patches(training_root, save_path, input_size, label_size, scale_factor)

    # read training data
    data, label = read_h5_file('./datasets/training_data_91_image_patches.h5')
    print(data.shape)
    print(label.shape)
