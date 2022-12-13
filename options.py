import argparse
import os
from utils import util
import torch

def str2bool(v):
    return v.lower() in ('true')

class Options():
    def __init__(self):
       self.parser = argparse.ArgumentParser()
       self.initialized = False

    def initialize(self):
        # Model configuration.
        self.parser.add_argument('--c1_dim', type=int, default=17, help='dimension of domain labels (AUs+age+gender)')
        self.parser.add_argument('--image_size', type=int, default=128, help='image resolution')
        self.parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
        self.parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
        self.parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
        self.parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
        self.parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

        # Training configuration.
        self.parser.add_argument('--dataset', type=str, default='AUs', choices=['AUs', 'CACD', 'Both'])
        self.parser.add_argument('--batch_size', type=int, default=25, help='mini-batch size')
        self.parser.add_argument('--n_threads_train', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--num_iters_validate', default=1, type=int, help='# batches to use when validating')
        self.parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self.parser.add_argument('--display_freq_s', type=int, default=300,help='frequency [s] of showing training results on screen')
        self.parser.add_argument('--save_latest_freq_s', type=int, default=3600, help='frequency of saving the latest results')

        self.parser.add_argument('--nepochs_no_decay', type=int, default=150, help='# of epochs at starting learning rate')
        self.parser.add_argument('--nepochs_decay', type=int, default=50, help='# of epochs to linearly decay learning rate to zero')

        self.parser.add_argument('--train_G_every_n_iterations', type=int, default=1, help='train G every n interations')
        self.parser.add_argument('--poses_g_sigma', type=float, default=0.06, help='initial learning rate for adam')
        self.parser.add_argument('--lr_G', type=float, default=0.00005, help='initial learning rate for G adam')
        self.parser.add_argument('--G_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self.parser.add_argument('--G_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        self.parser.add_argument('--lr_D', type=float, default=0.00005, help='initial learning rate for D adam')
        self.parser.add_argument('--D_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self.parser.add_argument('--D_adam_b2', type=float, default=0.999, help='beta2 for D adam')
        self.parser.add_argument('--lambda_D_prob', type=float, default=1, help='lambda for real/fake discriminator loss')
        self.parser.add_argument('--lambda_D_cond', type=float, default=160, help='lambda for condition discriminator loss')
        self.parser.add_argument('--lambda_cyc', type=float, default=10, help='lambda cycle loss')
        self.parser.add_argument('--lambda_mask', type=float, default=0.2, help='lambda mask loss')
        self.parser.add_argument('--lambda_D_gp', type=float, default=10, help='lambda gradient penalty loss')
        self.parser.add_argument('--lambda_mask_smooth', type=float, default=1e-5, help='lambda mask smooth loss')
        self.parser.add_argument('--lambda_pix', type=float, default=0, help='lambda mask smooth loss')

        # Test configuration.
        self.parser.add_argument('--output_dir', type=str, default='MDGAN-pg/sample-cfee-PG(6-13)',   help='out put path of generated image')
        self.parser.add_argument('--buffer_dir', type=str, default='MDGAN-pg/buffer-cfee-pg(6-13)', help='out put path of generated image')
        self.parser.add_argument('--generated_aus_file', type=str, default='des_labels_for_generation.pkl',
                                 help='file containing samples aus')

        # Miscellaneous.
        self.parser.add_argument('--num_workers', type=int, default=1)
        self.parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
        self.parser.add_argument('--use_tensorboard', type=str2bool, default=True)

        self.parser.add_argument('--load_epoch', type=int, default=-1,
                                 help='which epoch to load? set to -1 to use latest cached model')
        self.parser.add_argument('--gpu_ids', type=str, default='6', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='MD-CFEE-pg-6-13',  help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--n_threads_test', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', default=True,   help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--do_saturate_mask', action="store_true", default=False,  help='do use mask_fake for mask_cyc')
        self.parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset or not')

        # Directories.
        self.parser.add_argument('--aus_root', type=str, default='/home/sh_jun/Datasets', help='path to dataset')
        self.parser.add_argument('--aus_file', type=str, default='meta-data/2021-6-13/CFEE_op_aus_12_4.pkl',
                                 help='file containing samples aus')
        self.parser.add_argument('--aus_train_ids_file', type=str, default='meta-data/2021-6-13/CFEE_train_list-8g-no-id-6-13.pkl',
                                 help='file containing train ids')
        self.parser.add_argument('--aus_test_ids_file', type=str, default='meta-data/2021-6-13/CFEE_test_list-8g-no-id-6-13.pkl',
                                 help='file containing test ids')
        self.parser.add_argument('--label_index_file', type=str, default='meta-data/2021-6-13/CFEE_op_label_name_dict(12-8).pkl',
                                 help='file containing test ids')

        # self.parser.add_argument('--aus_root', type=str, default='D:/Datasets/CACD/', help='path to dataset')
        # self.parser.add_argument('--aus_file', type=str, default='2021-6-13/CFEE_op_aus_12_4.pkl', help='file containing samples aus')
        # self.parser.add_argument('--aus_train_ids_file', type=str, default='2021-6-13/CFEE_train_list-8g-no-id-6-13.pkl',
        #                          help='file containing train ids')
        # self.parser.add_argument('--aus_test_ids_file', type=str, default='2021-6-13/CFEE_test_list-8g-no-id-6-13.pkl',
        #                          help='file containing test ids')
        # self.parser.add_argument('--label_index_file', type=str, default='2021-6-13/CFEE_op_label_name_dict(12-8).pkl',
        #                          help='file containing test ids')


        # self.parser.add_argument('--aus_root', type=str, default='/home/sh_jun/Datasets', help='path to dataset')
        # self.parser.add_argument('--aus_file', type=str,
        #                          default='meta-data/2020-12-8/EM_aus_dict(75k).pkl', help='file containing samples aus')
        # self.parser.add_argument('--aus_train_ids_file', type=str,
        #                          default='meta-data/2020-12-8/EM_train_id(75k).pkl',  help='file containing train ids')
        # self.parser.add_argument('--aus_test_ids_file', type=str,
        #                          default='meta-data/2020-12-8/EM_test_id(75k).pkl', help='file containing test ids')

        # self.parser.add_argument('--aus_root', type=str, default='/home/sh_jun/Datasets', help='path to dataset')
        # self.parser.add_argument('--aus_file', type=str,
        #                          default='meta-data/2020-12-4/CFEE_op_aus_12_4.pkl', help='file containing samples aus')
        # self.parser.add_argument('--aus_train_ids_file', type=str,
        #                          default='meta-data/2020-12-30/CFEE_op_cls_train_id(12-30)-7g.pkl',
        #                          help='file containing train ids')
        # self.parser.add_argument('--aus_test_ids_file', type=str,
        #                          default='meta-data/2020-12-30/CFEE_op_cls_test_id(12-30)-7g.pkl',
        #                          help='file containing test ids')
        # self.parser.add_argument('--label_index_file', type=str,
        #                          default='meta-data/2020-12-8/CFEE_op_label_name_dict(12-8).pkl',
        #                          help='file containing test ids')

        # self.parser.add_argument('--aus_root', type=str, default='/home/sh_jun/Datasets', help='path to dataset')
        # self.parser.add_argument('--aus_file', type=str, default='meta-data/2020-12-4/CFEE_op_aus_12_4.pkl',
        #                          help='file containing samples aus')
        # self.parser.add_argument('--aus_train_ids_file', type=str,
        #                          default='meta-data/2020-12-8/CFEE_op_cls_train_id(12-8).pkl',
        #                          help='file containing train ids')
        # self.parser.add_argument('--aus_test_ids_file', type=str,
        #                          default='meta-data/2020-12-8/CFEE_op_cls_test_id(12-8).pkl',
        #                          help='file containing test ids')
        # self.parser.add_argument('--label_index_file', type=str,
        #                          default='meta-data/2020-12-8/CFEE_op_label_name_dict(12-8).pkl',
        #                          help='file containing test ids')

        # self.parser.add_argument('--aus_train_ids_file', type=str, default='meta-data/2020-8-27/CR_train_id(9-4).pkl',
        #                          help='file containing train ids')
        # self.parser.add_argument('--aus_test_ids_file', type=str, default='meta-data/2020-8-27/CR_test_id(9-4).pkl',
        #                          help='file containing test ids')



        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.config =self.parser.parse_args()

        # # set is train or set
        # self.config.is_train = self.is_train

        # set and check load_epoch
        self.set_and_check_load_epoch()

        # get and set gpus
        self.get_set_gpus()

        args = vars(self.config)

        # print in terminal args
        self._print(args)

        # save args to file
        self.save(args)

        return self.config

    def set_and_check_load_epoch(self):
        models_dir = os.path.join(self.config.checkpoints_dir,self.config.name)
        if os.path.exists(models_dir):
            if self.config.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self.config.load_epoch = load_epoch
            else:
                 found = False
                 for file in os.listdir(models_dir):
                     if file.startswith("net_epoch_"):
                         found = int(file.split('_')[2]) == self.config.load_epoch
                         if found:  break
                 assert found, 'Model for epoch %i not found' % self.config.load_epoch
        else:
            assert self.config.load_epoch < 1, 'Model for epoch %i not found' % self.config.load_epoch
            self.config.load_epoch = 0

    def get_set_gpus(self):
        # get gpu ids
        str_ids =self.config.gpu_ids.split(',')
        self.config.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
               self.config.gpu_ids.append(id)

        # set gpu ids
        if len(self.config.gpu_ids) > 0:
            torch.cuda.set_device(self.config.gpu_ids[0])

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def save(self, args):
        expr_dir = os.path.join(self.config.checkpoints_dir,self.config.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % (self.config.mode))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
