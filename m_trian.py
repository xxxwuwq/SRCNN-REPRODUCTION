import time
from m_nets import *
from m_utils import *

# GPU设备使用
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

learning_rate = 0.00001
train_num = 800000000

X = tf.placeholder('float32', [None, 33, 33, 1])
Y = tf.placeholder('float32', [None, 21, 21, 1])
inference = srcnn(X, pad='VALID', name='srcnn')
cost = tf.reduce_mean(tf.square(Y-inference))
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

batch_size = 64
datasets_path = '../datasets/training/training_data_91_image_patches.h5'
data, label = read_h5_file(datasets_path)
batch = data_iterator(data, label, batch_size)
ckpts_dir = './ckpts/'
log = open('./logs/training.log', 'a+')
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    print('start session')
    if ckpt and ckpt.model_checkpoint_path:
        print('load model...')
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(0, train_num):
        batch_data, batch_label = batch.__next__()
        loss, _ = sess.run([cost, optimizer], feed_dict={X: batch_data, Y: batch_label})

        if i % 450 == 0:
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            log_info = format_time + ' epoch %d and loss %f'%(i // 450, loss)
            log.writelines(log_info + '\n')
            print(log_info)

        if i % 5000 == 0:
            saver.save(sess, ckpts_dir + 'model_2x.ckpt', i)





