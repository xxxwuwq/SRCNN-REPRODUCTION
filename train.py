import time
from networks import SRCNN
from utils import Tools, set_gpu
from configs import Config
import tensorflow as tf
import os
from tqdm import tqdm
set_gpu(0)


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def train(cfg, data_loder, test_data):
    TRAIN_INPUTS = tf.placeholder('float32', [None, 33, 33, 1])
    TRAIN_LABELS = tf.placeholder('float32', [None, 21, 21, 1])

    VAL_INPUTS = tf.placeholder('float32', [None, None, None, 1])
    VAL_LABELS = tf.placeholder('float32', [None, None, None, 1])

    model = SRCNN()
    train_inference = model(TRAIN_INPUTS, padding='VALID', name='log')
    val_inference = model(VAL_INPUTS, padding='SAME', name='log')

    train_loss = tf.losses.mean_squared_error(TRAIN_LABELS / 255.0, train_inference)
    val_loss = tf.losses.mean_squared_error(VAL_LABELS / 255.0, val_inference)
    train_op = tf.train.AdamOptimizer(cfg.lr).minimize(train_loss)

    # mse = tf.reduce_mean(tf.square(val_inference - VAL_LABELS))
    # psnr = 20 * log10(255 / tf.sqrt(mse))
    #
    writer = tf.summary.FileWriter(cfg.events_dir, tf.get_default_graph())
    tf.summary.scalar('train_loss', train_loss, collections=None)
    # tf.summary.scalar('val_loss', val_loss, collections=None)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    train_log = open(os.path.join(cfg.log_dir, 'train.log'), mode='a+', encoding='utf-8')
    val_log = open(os.path.join(cfg.log_dir, 'val.log'), mode='a+', encoding='utf-8')
    saver = tf.train.Saver(max_to_keep=cfg.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(cfg.ckpt_dir)

    if ckpt and ckpt.model_checkpoint_path:
        print('load model...')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Finished!')

    merge_summary = tf.summary.merge_all()
    for i in range(0, cfg.total_iters):
        batch_data, batch_label = data_loder.__next__()

        train_summary, loss, _, inference = sess.run([merge_summary, train_loss, train_op, train_inference], feed_dict={TRAIN_INPUTS: batch_data, TRAIN_LABELS: batch_label})

        if i % cfg.train_print == 0:
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            log_info = format_time + ' iters:%d, loss:%.6f' % (i, loss)
            train_log.writelines(log_info + '\n')
            print(log_info)

        if i % cfg.val_print == 0 and i != 0:
            saver.save(sess, cfg.ckpt_dir + 'model_%dx.ckpt' % cfg.scale_factor, 0)
            for j in range(len(test_data)):

                loss, inference = sess.run([val_loss, val_inference], feed_dict={VAL_INPUTS: test_data[j][1], VAL_LABELS: test_data[j][2]})
                inference[inference > 1.0] = 1.0
                inference[inference < 0.0] = 0.0
                inference = inference * 255.0

                metric = tool.psnr(inference, test_data[j][2])
                format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                log_info = format_time + ' ' + 'iters:%d, img:%d, loss:%.6f, psnr:%.6f' % (i, j, loss, metric)
                print(log_info)
                val_log.write(log_info + '\n')
        writer.add_summary(train_summary, i)
    writer.close()


if __name__ == '__main__':
    cfg = Config('SRCNN')
    tool = Tools()
    batch_size = 64
    # train data
    datasets_path = './datasets/training_91_image_patches.h5'
    data, label = tool.read_h5_file(datasets_path)
    data_loder = tool.data_iterator(data, label, batch_size)

    # val data
    path = './datasets/Test/Set5'
    test_data = tool.read_test_data(path, cfg)

    train(cfg, data_loder, test_data)