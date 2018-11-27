from networks import *
from utils import *
from configs import *
from skimage.measure import compare_psnr,compare_mse

np.set_printoptions(threshold=np.inf)


def test():
    set_gpu(2)
    cfig = ConfigFactory('srcnn')
    image = cv.imread('./baby_GT.bmp')
    image = modcrop(image, cfig.scale_factor)
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    (hei, wid, cha) = image.shape
    im_label = image[:, :, 0]
    im_input = cv.resize(im_label, (0, 0), fx=1.0 / cfig.scale_factor, fy=1.0 / cfig.scale_factor, interpolation=cv.INTER_CUBIC)
    im_input = cv.resize(im_input, (0, 0), fx=cfig.scale_factor, fy=cfig.scale_factor, interpolation=cv.INTER_CUBIC).astype(np.float32)

    # input and ground truth of network
    im_Y_label = im_label.reshape((1, hei, wid, 1)).astype(np.float32) / 255.0
    im_Y_input = im_input.reshape((1, hei, wid, 1)).astype(np.float32) / 255.0
    # inference
    inference = srcnn(im_Y_input, padding='SAME', name='srcnn')
    loss = tf.losses.mean_squared_error(im_Y_label, inference)

    # start session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=cfig.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(cfig.ckpt_router)

    if ckpt and ckpt.model_checkpoint_path:
        print('load model', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        test_loss, test_inference = sess.run([loss, inference])

        test_inference[test_inference > 1.0] = 1.0
        test_inference[test_inference < 0.0] = 0.0
        test_inference = test_inference * 255.0

        # test_inference and label should be float32 or compute diff will overflow
        test_inference = test_inference.reshape((test_inference.shape[1], test_inference.shape[2])).astype(dtype=np.uint8)
        ori_modcrop_image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
        cv.imwrite('./test_save/ori_modcrop.bmp', ori_modcrop_image)
        image[:, :, 0] = test_inference
        sr = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
        cv.imwrite('./test_save/srcnn_rescon.bmp', sr)

        test_inference = test_inference.astype(np.float32)
        im_Y_label = im_Y_label[0, :, :, 0] * 255.
        err = compare_mse(im_Y_label, test_inference)
        psnr_metric = 10 * np.log10((255 ** 2) / err)
        print(psnr_metric)


if __name__ == '__main__':
    test()


