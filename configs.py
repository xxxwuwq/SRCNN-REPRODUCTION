import os
class Config:

    def __init__(self, name='srcnn'):
        self.name = name
        self.batch_size = 64
        self.lr = 1e-6
        self.lr_decay = 0.9
        self.scale_factor = 3
        self.total_iters = 400000000
        self.max_ckpt_keep = 2000
        self.train_print = 5000
        self.val_print = 50000
        self.ckpt_dir = os.path.join('./train_log', 'ckpts/')
        self.log_dir = os.path.join('./train_log', 'logs/')
        self.events_dir = os.path.join('./train_log', 'events/')

        # training log route
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # model saver route
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        if not os.path.exists(self.events_dir):
            os.makedirs(self.events_dir)

    def display_configs(self):
        msg = '''
        ------------ info of %s model -------------------
        batch size              : %s
        learing rate            : %f
        learing rate decay      : %f
        scale factor:           : %d
        iter num                : %s
        max ckpt keep           : %s
        ckpt router             : %s
        log router              : %s
        ------------------------------------------------
        ''' % (self.name, self.batch_size, self.lr, self.lr_decay, self.scale_factor, self.total_iters, self.max_ckpt_keep, self.ckpt_router, self.log_router)
        print(msg)
        return msg


if __name__ == '__main__':
    configs = Config()
    configs.display_configs()
