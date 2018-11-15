import time
from m_nets import *
from m_utils import *
import cv2 as cv

np.set_printoptions(threshold=np.inf)
# GPU设备使用
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

scale = 2
# image = cv.imread('./butterfly_GT.bmp')
image = cv.imread('./baby_GT.bmp')
image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
image = image[:, :, 0:3]
im_label = modcrop(image, 2)
label = im_label[:, :, 0].astype(np.float32)

(hei, wid, _) = im_label.shape
im_input = cv.resize(im_label, (0, 0), fx=1.0/scale, fy=1.0/scale, interpolation=cv.INTER_CUBIC)
im_input = cv.resize(im_input, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

input_YChannel = im_input[:, :, 0].reshape((1, hei, wid, 1)) / 255.0
input_YChannel = input_YChannel.astype(np.float32)
label_YChannel = im_label[:, :, 0].reshape((1, hei, wid, 1)) / 255.0
print(label_YChannel.shape)

inference = srcnn(input_YChannel, pad='SAME', name='srcnn')
cost = tf.reduce_mean(tf.square(label_YChannel - inference))

ckpts_dir = './ckpts/'
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    print('start session')
    if ckpt and ckpt.model_checkpoint_path:
        print('load model...')
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    mse, output = sess.run([cost, inference])

    output[output > 1.0] = 1.0
    output[output < 0.0] = 0.0
    output = output * 255.0

    # output and label should be float32 or compute diff will overflow
    output = output.reshape((output.shape[1], output.shape[2]))
    output = output.astype(dtype=np.uint8)
    im_input[:, :, 0] = output
    sr = cv.cvtColor(im_input, cv.COLOR_YCrCb2BGR)
    cv.imwrite('./baby_srcnn.bmp', sr)
    # cv.imwrite('./butterfly_srcnn.bmp', sr)

    output = output.astype(np.float32)
    imdff = output[6:506, 6:506] - label[6:506, 6:506]
    # imdff = output - label
    rmse = np.sqrt(np.mean(np.power(imdff, 2)))
    psnr = 20. * np.log10(255. / rmse)
    print(psnr)




