import numpy as np
import cv2 as cv
import os
import h5py

np.set_printoptions(threshold=np.inf)


class Tools():
    def __init__(self):
        pass

    def psnr(self, img1, img2):
        """
        compute the psnr
        :param img1: img1
        :param img2: img2
        :return:
        """
        diff = np.abs(img1 - img2)
        mse = np.square(diff).mean()
        psnr = 20 * np.log10(255 / np.sqrt(mse))
        return psnr

    def make_train_h5(self, training_root, save_path, input_size=33, label_size=21, scale_factor=3):
        '''
        make training data(h5 file)
        :param training_root: the dir of traning dataset
        :param save_path: name of ht file
        :param input_size: the input img size for training (set to be 33*33, default)
        :param label_size: the label size for training (set to be 21*21, default)
        :param scale_factor: (set to be 3, default)
        :return:
        '''
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
                im_label = self.__modcrop(image, scale_factor)
                (hei, wid, _) = im_label.shape
                # scale to 1 / s
                im_input = cv.resize(im_label, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor, interpolation=cv.INTER_CUBIC)
                # scale to s
                im_input = cv.resize(im_input, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)

                # low resolution for input
                im_input = im_input.astype('float32')
                # high resolution for label
                im_label = im_label.astype('float32')

                input_images.append(im_input)
                label_images.append(im_label)
                for x in range(0, hei - input_size + 1, stride):
                    for y in range(0, wid - input_size + 1, stride):
                        sub_im_input = im_input[x:x + input_size, y:y + input_size, 0]
                        sub_im_label = im_label[x + padding:x + padding + label_size,
                                       y + padding: y + padding + label_size,
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

    def read_test_data(self, path, cfg):
        flist = os.listdir(path)
        data = []
        for f in flist:
            print(f)
            img = cv.imread(os.path.join(path, f))
            img = self.__modcrop(img, cfg.scale_factor)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

            (h, w, c) = img.shape
            im_input = cv.resize(img[:, :, 0], (0, 0), fx=1.0 / cfg.scale_factor, fy=1.0 / cfg.scale_factor,
                                 interpolation=cv.INTER_CUBIC)
            im_input = cv.resize(im_input, (0, 0), fx=cfg.scale_factor, fy=cfg.scale_factor,
                                 interpolation=cv.INTER_CUBIC).astype(np.float32)
            Y_input = im_input.reshape((1, h, w, 1)).astype(np.float32)
            Y_label = img[:, :, 0].reshape((1, h, w, 1)).astype(np.float32)

            data.append([img, Y_input, Y_label])
        return data

    def read_h5_file(self, path):
        """
        read data from h5 file
        :param path:
        :return:
        """
        with h5py.File(path, 'r') as hf:
            hf_data = hf.get('input')
            data = np.array(hf_data)
            hf_label = hf.get('label')
            label = np.array(hf_label)
            return data, label

    def data_iterator(self, data, label, batch_size):
        """
        training data generator
        :param data: img data
        :param label: label data
        :param batch_size: mini-batch size
        :return:
        """
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

    def __search(self, dirname):
        '''
        the images suffixed with BMP on training and testing(set 5 and set14) dataset
        :param dirname: the dataset root dir
        :return: the name of image files
        '''
        train_name = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filenname = os.path.join(dirname, filename)
            ext = os.path.splitext(full_filenname)[-1]
            if ext == '.bmp':
                train_name.append(full_filenname)
        return train_name

    def __modcrop(self, imgs, modulo):
        '''
        crop the image to make the H and W be integer multiples of 3
        :param imgs:
        :param modulo:
        :return:
        '''
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


def set_gpu(gpu=0):
    """
    the gpu used setting
    :param gpu: gpu id
    :return:
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


if __name__ == '__main__':
    # 1. generate trianing data
    training_root = r'./datasets/Train'
    save_path = './datasets/training_91_image_patches.h5'
    input_size = 33
    label_size = 21
    scale_factor = 3
    tool = Tools()
    tool.make_train_h5(training_root, save_path, input_size, label_size, scale_factor)
    #
    # # 2. read training data
    # data, label = tool.read_h5_file('./datasets/training_91_image_patches.h5')
    # print(data.shape)
    # print(label.shape)
    from configs import Config
    cfg = Config('SRCNN')
    tool = Tools()
    batch_size = 64
    datasets_path = './datasets/training_91_image_patches.h5'
    data, label = tool.read_h5_file(datasets_path)
    data_loder = tool.data_iterator(data, label, batch_size)
    path = './datasets/Test/Set5'
    test_data = tool.read_test_data(path, cfg)
    # print(test_data[0][0])
    # print(test_data[0][1])
    # print(test_data[0][2][0, :, :, 0])
    # img, label = data_loder.__next__()
    # print(label)