import time
from networks import *
from utils import *
from configs import *


def train():
    # gpu setting
    set_gpu(0)
    cfig = ConfigFactory('srcnn')
    INPUTS_PALCEHOLDER = tf.placeholder('float32', [None, 33, 33, 1])
    GROUNDTRUTH_PALCEHOLDER = tf.placeholder('float32', [None, 21, 21, 1])
    inference = srcnn(INPUTS_PALCEHOLDER, padding='VALID', name='srcnn')

    loss = tf.losses.mean_squared_error(GROUNDTRUTH_PALCEHOLDER, inference)
    train_op = tf.train.AdamOptimizer(cfig.lr).minimize(loss)

    batch_size = 64
    datasets_path = './datasets/training_data_91_image_patches.h5'
    data, label = read_h5_file(datasets_path)
    batch = data_iterator(data, label, batch_size)

    init = tf.global_variables_initializer()
    # start session
    sess = tf.InteractiveSession()
    sess.run(init)

    file_path = cfig.log_router
    # training log route
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # model saver route
    if not os.path.exists(cfig.ckpt_router):
        os.makedirs(cfig.ckpt_router)
    log = open(cfig.log_router + cfig.name + r'_training.logs', mode='a+', encoding='utf-8')
    saver = tf.train.Saver(max_to_keep=cfig.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(cfig.ckpt_router)

    if ckpt and ckpt.model_checkpoint_path:
        print('load model')
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(0, cfig.total_iters):
        batch_data, batch_label = batch.__next__()
        train_loss, _ = sess.run([loss, train_op],
                                 feed_dict={INPUTS_PALCEHOLDER: batch_data, GROUNDTRUTH_PALCEHOLDER: batch_label})

        if i % 450 == 0:
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            log_info = format_time + ' epoch %d and loss %f' % (i // 450, train_loss)
            log.writelines(log_info + '\n')
            print(log_info)

        if i % 5000 == 0:
            saver.save(sess, cfig.ckpt_router + 'model_3x.ckpt', i)


if __name__ == '__main__':
    train()