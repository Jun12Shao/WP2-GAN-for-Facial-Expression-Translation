import torch
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim import lr_scheduler
import utils.util as util
import utils.plots as plot_utils
from models.model import Generator, Discriminator0, Discriminator1, Discriminator2
import os
import numpy as np
import glob


# from models.vggface import VGGFace
# import torch.nn.functional as F


class BaseModel(object):

    def __init__(self, config):
        self.name = 'BaseModel'

        self.config = config
        self.gpu_ids = config.gpu_ids

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(config.checkpoints_dir, config.name)

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    # used in test time, no backprop
    def test(self):
        assert False, "test not implemented"

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_current_visuals(self):
        return {}

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self.save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        device = torch.device("cuda")
        optimizer.load_state_dict(torch.load(load_path, map_location=device))
        print('loaded optimizer: %s' % load_path)

    def save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print('saved net: %s' % save_path)

    def load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self.save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        device = torch.device("cuda")
        network.load_state_dict(torch.load(load_path, map_location=device))
        print('loaded net: %s' % load_path)

    def update_learning_rate(self):
        pass

    def print_network(self, network, name):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print(name)
        print('Total number of parameters: %d' % num_params)

    def _get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler


########################################################################################################
class AE_GAN(BaseModel):
    def __init__(self, config):
        super(AE_GAN, self).__init__(config)
        self.name = 'AE_GAN'
        self.config = config
        if config.mode == 'train':
            self.is_train = True
        else:
            self.is_train = False

        if self.config.dataset == 'AUs':
            self.cond_nc = self.config.c1_dim
        else:
            raise ("Wrong input of dataset_name.")

        # create networks
        self.init_create_networks()

        # init train variables
        if self.is_train:
            self.init_train_vars()  ##将网络的各种参数赋予网络

        # load networks and optimizers
        if not self.is_train or self.config.load_epoch > 0:
            self.load()

        # prefetch variables
        self.init_prefetch_inputs()

        # init
        self.init_losses()

    def init_create_networks(self):
        # generator network
        self.G_A = self.create_generator()
        self.print_network(self.G_A, 'G_A')  #############

        self.G_A.init_weights()
        if len(self.gpu_ids) > 1:
            self.G_A = torch.nn.DataParallel(self.G_A, device_ids=self.gpu_ids)
        self.G_A.cuda()

        # generator network
        self.G_B = self.create_generator()
        self.print_network(self.G_B, 'G_B')  #############

        self.G_B.init_weights()
        if len(self.gpu_ids) > 1:
            self.G_B = torch.nn.DataParallel(self.G_B, device_ids=self.gpu_ids)
        self.G_B.cuda()

        # discriminator network
        self.D0 = self.create_discriminator('D0')
        self.print_network(self.D0, 'D0')  ####################

        self.D0.init_weights()
        if len(self.gpu_ids) > 1:
            self.D0 = torch.nn.DataParallel(self.D0, device_ids=self.gpu_ids)
        self.D0.cuda()

        # discriminator network
        self.D1 = self.create_discriminator('D1')
        self.print_network(self.D1, 'D1')  ####################

        self.D1.init_weights()
        if len(self.gpu_ids) > 1:
            self.D1 = torch.nn.DataParallel(self.D1, device_ids=self.gpu_ids)
        self.D1.cuda()

        # discriminator network
        self.D2 = self.create_discriminator('D2')
        self.print_network(self.D2, 'D2')  ####################

        self.D2.init_weights()
        if len(self.gpu_ids) > 1:
            self.D2 = torch.nn.DataParallel(self.D2, device_ids=self.gpu_ids)
        self.D2.cuda()


    def create_generator(self):
        return Generator(c_dim=self.cond_nc)

    def create_discriminator(self, label):
        if label == 'D0':
            return Discriminator0(c_dim=self.cond_nc)
        elif label == 'D1':
            return Discriminator1(c_dim=self.cond_nc)
        elif label == 'D2':
            return Discriminator2(c_dim=self.cond_nc)

    def init_train_vars(self):
        self.current_lr_G = self.config.lr_G
        self.current_lr_D = self.config.lr_D

        # initialize optimizers
        # initialize optimizers
        self.optimizer_G_A = torch.optim.Adam(self.G_A.parameters(), lr=self.current_lr_G,
                                              betas=[self.config.G_adam_b1, self.config.G_adam_b2])

        self.optimizer_G_B = torch.optim.Adam(self.G_B.parameters(), lr=self.current_lr_G,
                                              betas=[self.config.G_adam_b1, self.config.G_adam_b2])

        self.optimizer_D0 = torch.optim.Adam(self.D0.parameters(), lr=self.current_lr_D,
                                             betas=[self.config.D_adam_b1, self.config.D_adam_b2])
        self.optimizer_D1 = torch.optim.Adam(self.D1.parameters(), lr=self.current_lr_D,
                                             betas=[self.config.D_adam_b1, self.config.D_adam_b2])
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=self.current_lr_D,
                                             betas=[self.config.D_adam_b1, self.config.D_adam_b2])

    def init_prefetch_inputs(self):
        self.input_real_img = self.Tensor(self.config.batch_size, 3, self.config.image_size,
                                          self.config.image_size)  ##construct a zero tensor of size ?*3*128*128
        # self.input_ref_img1 = self.Tensor(self.config.batch_size, 3, self.config.image_size,
        #                                   self.config.image_size)  ##construct a zero tensor of size ?*3*128*128
        # self.input_ref_img2 = self.Tensor(self.config.batch_size, 3, self.config.image_size,
        #                                   self.config.image_size)  ##construct a zero tensor of size ?*3*128*128

        self.input_real_cond = self.Tensor(self.config.batch_size, self.cond_nc)  ##construct a zero tensor of size ?*17
        self.input_desired_cond = self.Tensor(self.config.batch_size, self.cond_nc)
        self.input_real_img_path = None
        self.input_real_cond_path = None

    def init_losses(self):
        # define loss functions
        self.criterion_cycle = torch.nn.L1Loss().cuda()  ## L1 loss used for identity loss
        self.criterion_D_cond = torch.nn.MSELoss().cuda()  ## L2 loss or MSE loss used for conditional loss
        self.criterion_D_age = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_idt = torch.nn.MSELoss().cuda()  ## L2 loss or MSE loss used for conditional loss

        # init losses G
        # self.loss_g_fake = Variable(self.Tensor([0]))
        # self.loss_g_cond = Variable(self.Tensor([0]))
        self.loss_g_cyc = Variable(self.Tensor([0]))
        self.loss_g_mask_1 = Variable(self.Tensor([0]))
        self.loss_g_mask_2 = Variable(self.Tensor([0]))
        # self.loss_g_idt = Variable(self.Tensor([0]))
        self.loss_g_masked_fake = Variable(self.Tensor([0]))
        self.loss_g_masked_cond = Variable(self.Tensor([0]))
        loss_g_masked_age = Variable(self.Tensor([0]))
        loss_g_masked_gender = Variable(self.Tensor([0]))
        self.loss_g_mask_1_smooth = Variable(self.Tensor([0]))
        self.loss_g_mask_2_smooth = Variable(self.Tensor([0]))
        # self.loss_rec_real_img_rgb = Variable(self.Tensor([0]))
        # self.loss_g_fake_imgs_smooth = Variable(self.Tensor([0]))
        # self.loss_g_unmasked_rgb = Variable(self.Tensor([0]))

        # init losses D
        self.loss_d_real = Variable(self.Tensor([0]))
        self.loss_d_cond = Variable(self.Tensor([0]))
        self.loss_d_fake = Variable(self.Tensor([0]))
        # self.loss_d_cond_age = Variable(self.Tensor([0]))
        # self.loss_d_cond_gender = Variable(self.Tensor([0]))
        self.loss_d_gp = Variable(self.Tensor([0]))

    def set_input(self, input):
        self.input_real_img.resize_(input['real_img'].size()).copy_(input['real_img'])
        # self.input_ref_img1.resize_(input['ref_img1'].size()).copy_(input['ref_img1'])
        # self.input_ref_img2.resize_(input['ref_img2'].size()).copy_(input['ref_img2'])

        self.input_real_cond.resize_(input['real_cond'].size()).copy_(input['real_cond'])
        self.input_desired_cond.resize_(input['desired_cond'].size()).copy_(input['desired_cond'])
        self.input_real_id = input['sample_id']
        self.input_real_img_path = input['real_img_path']

        if len(self.gpu_ids) > 0:
            self.input_real_img = self.input_real_img.cuda(self.gpu_ids[0])
            self.input_real_cond = self.input_real_cond.cuda(self.gpu_ids[0])
            self.input_desired_cond = self.input_desired_cond.cuda(self.gpu_ids[0])

    def set_train(self):
        self.G_A.train()  ## Sets the module in training mode
        self.G_B.train()  ## Sets the module in training mode
        self.D0.train()  ## Sets the module in training mode
        self.D1.train()  ## Sets the module in training mode
        self.D2.train()  ## Sets the module in training mode
        self.is_train = True

    def set_eval(self):
        self.G_A.eval()  ## Sets the module in evaluating mode
        self.G_B.eval()  ## Sets the module in evaluating mode
        self.is_train = False

    # get image paths
    def get_image_paths(self):
        return OrderedDict([('real_img', self.input_real_img_path)])

    def forward(self, keep_data_for_visuals=False,  return_estimates=True):  ##当直接调用GANimation model时使用， 目前该function 未被调用
        if not self.is_train:
            # convert tensor to variables
            real_img = Variable(self.input_real_img)
            real_cond = Variable(self.input_real_cond)
            desired_cond = Variable(self.input_desired_cond)

            res_cond = desired_cond - real_cond
            real_img1 = real_img

            for p in [0.3, 0.6, 1.0]:
                des_cond = real_cond + p * res_cond
                # generate fake images
                fake_imgs, fake_img_mask = self.G_A.forward(real_img1, des_cond)
                fake_img_mask = self.do_if_necessary_saturate_mask(fake_img_mask, saturate=self.config.do_saturate_mask)
                fake_imgs_masked = fake_img_mask * real_img + (1 - fake_img_mask) * fake_imgs

                real_img1 = fake_imgs_masked.detach()



            imgs = None
            data = None
            if return_estimates:
                # generate images
                im_real_img = util.tensor2im(real_img.data)

                im_fake_imgs_masked = util.tensor2im(fake_imgs_masked.data)


                im_real_img_batch = util.tensor2im(real_img.data, idx=-1, nrows=1)
                im_fake_imgs_masked_batch = util.tensor2im(fake_imgs_masked.data, idx=-1, nrows=1)

                imgs = OrderedDict([('real_img_batch', im_real_img_batch),
                                    ('fake_imgs_masked', im_fake_imgs_masked),
                                    ('fake_imgs_masked_batch', im_fake_imgs_masked_batch) ])


            return imgs

    def optimize_parameters(self, train_generator=True,
                            keep_data_for_visuals=False):  ## the mian function for the learning
        if self.is_train:
            # convert tensor to variables
            self.B = self.input_real_img.size(0)
            self.real_img = Variable(self.input_real_img)
            # self.ref_img1=  Variable(self.input_ref_img1)
            # self.ref_img2 = Variable(self.input_ref_img2)

            self.real_cond = Variable(self.input_real_cond)
            self.desired_cond = Variable(self.input_desired_cond)

            real_cond = self.real_cond
            res_cond = self.desired_cond - self.real_cond
            real_img1 = self.real_img
            real_img2 = self.real_img
            for p in [0.3, 0.6, 1.0]:
                desired_cond = self.real_cond + p * res_cond

                # train D
                # train D
                loss_D, fake_imgs_masked1, fake_imgs_masked2 = self.forward_D(real_img1, real_img2, desired_cond)

                self.optimizer_D1.zero_grad()
                self.optimizer_D2.zero_grad()
                self.optimizer_D0.zero_grad()

                loss_D.backward()
                self.optimizer_D1.step()
                self.optimizer_D2.step()
                self.optimizer_D0.step()

                loss_D_gp0 = self.gradinet_penalty_D(self.D0, fake_imgs_masked1)
                self.optimizer_D0.zero_grad()
                loss_D_gp0.backward()
                self.optimizer_D0.step()

                loss_D_gp1 = self.gradinet_penalty_D(self.D1, fake_imgs_masked1)
                self.optimizer_D1.zero_grad()
                loss_D_gp1.backward()
                self.optimizer_D1.step()

                loss_D_gp2 = self.gradinet_penalty_D(self.D2, fake_imgs_masked1)
                self.optimizer_D2.zero_grad()
                loss_D_gp2.backward()
                self.optimizer_D2.step()

                self.loss_d_gp = loss_D_gp0 + loss_D_gp1 + loss_D_gp2

                # train G
                if train_generator:
                    loss_G = self.forward_G(real_img1, real_img2, real_cond, desired_cond)
                    self.optimizer_G_A.zero_grad()
                    self.optimizer_G_B.zero_grad()
                    loss_G.backward()
                    self.optimizer_G_A.step()
                    self.optimizer_G_B.step()

                real_cond = desired_cond
                real_img1 = fake_imgs_masked1
                real_img2 = fake_imgs_masked2


    def forward_G(self,real_img1,real_img2,real_cond,desired_cond):  ##calculate the loss_G
        # generate fake images
        fake_imgs1, fake_img_mask1 = self.G_A.forward(real_img1, desired_cond)
        fake_img_mask1 = self.do_if_necessary_saturate_mask(fake_img_mask1, saturate=self.config.do_saturate_mask)
        fake_imgs_masked1 = fake_img_mask1 * self.real_img + (1 - fake_img_mask1) * fake_imgs1

        # G(G(Ic1,c2), c1)
        rec_real_img_rgb1, rec_real_img_mask1 = self.G_B.forward(fake_imgs_masked1, real_cond)
        rec_real_img_mask1 = self.do_if_necessary_saturate_mask(rec_real_img_mask1, saturate=self.config.do_saturate_mask)
        rec_real_imgs1 = rec_real_img_mask1 * fake_imgs_masked1 + (1 - rec_real_img_mask1) * rec_real_img_rgb1

        # generate fake images
        fake_imgs2, fake_img_mask2 = self.G_B.forward(real_img2, desired_cond)
        fake_img_mask2 = self.do_if_necessary_saturate_mask(fake_img_mask2, saturate=self.config.do_saturate_mask)
        fake_imgs_masked2 = fake_img_mask2 * self.real_img + (1 - fake_img_mask2) * fake_imgs2

        # G(G(Ic1,c2), c1)
        rec_real_img_rgb2, rec_real_img_mask2 = self.G_A.forward(fake_imgs_masked2, real_cond)
        rec_real_img_mask2 = self.do_if_necessary_saturate_mask(rec_real_img_mask2, saturate=self.config.do_saturate_mask)
        rec_real_imgs2 = rec_real_img_mask2 * fake_imgs_masked2 + (1 - rec_real_img_mask2) * rec_real_img_rgb2


        # D(G(Ic1, c2)*M) masked

        d_fake_desired_img_masked_prob1_0, d_fake_desired_img_masked_cond1_0 = self.D0.forward(fake_imgs_masked1)
        d_fake_desired_img_masked_prob1_1, d_fake_desired_img_masked_cond1_1, = self.D1.forward(fake_imgs_masked1)
        d_fake_desired_img_masked_prob1_2, d_fake_desired_img_masked_cond1_2, = self.D2.forward(fake_imgs_masked1)

        d_fake_desired_img_masked_prob2_0, d_fake_desired_img_masked_cond2_0 = self.D0.forward(fake_imgs_masked2)
        d_fake_desired_img_masked_prob2_1, d_fake_desired_img_masked_cond2_1, = self.D1.forward(fake_imgs_masked2)
        d_fake_desired_img_masked_prob2_2, d_fake_desired_img_masked_cond2_2, = self.D2.forward(fake_imgs_masked2)


        self.loss_g_masked_fake = (self.compute_loss_D(d_fake_desired_img_masked_prob1_0, True) +
                                   self.compute_loss_D(d_fake_desired_img_masked_prob1_1, True) +
                                   self.compute_loss_D(d_fake_desired_img_masked_prob1_2, True)+
                                   self.compute_loss_D(d_fake_desired_img_masked_prob2_0, True) +
                                   self.compute_loss_D(d_fake_desired_img_masked_prob2_1, True) +
                                   self.compute_loss_D(d_fake_desired_img_masked_prob2_2, True)
                                   ) /6 * self.config.lambda_D_prob
        self.loss_g_masked_cond = (self.criterion_D_cond(d_fake_desired_img_masked_cond1_0, desired_cond) +
                                   self.criterion_D_cond(d_fake_desired_img_masked_cond1_1, desired_cond) +
                                   self.criterion_D_cond(d_fake_desired_img_masked_cond1_2, desired_cond)+
                                   self.criterion_D_cond(d_fake_desired_img_masked_cond2_0, desired_cond) +
                                  self.criterion_D_cond(d_fake_desired_img_masked_cond2_1, desired_cond) +
                                  self.criterion_D_cond(d_fake_desired_img_masked_cond2_2, desired_cond)
                                   ) / 6 * self.config.lambda_D_cond

        # l_cyc(G(G(Ic1,c2), c1)*M)
        self.loss_g_cyc = (self.criterion_cycle(rec_real_imgs1, real_img1)+self.criterion_cycle(rec_real_imgs2, real_img2))/2 * self.config.lambda_cyc
        self.loss_g_sm = self.criterion_cycle(fake_imgs_masked1, fake_imgs_masked2)*1

        # loss mask
        self.loss_g_mask_1 = (torch.mean(fake_img_mask1) + torch.mean(fake_img_mask2)) / 2 * self.config.lambda_mask
        self.loss_g_mask_2 = (torch.mean(rec_real_img_mask1) + torch.mean(rec_real_img_mask2)) / 2 * self.config.lambda_mask
        self.loss_g_mask_1_smooth = (self.compute_loss_smooth(fake_img_mask1) + self.compute_loss_smooth(fake_img_mask2)) / 2 * self.config.lambda_mask_smooth
        self.loss_g_mask_2_smooth = (self.compute_loss_smooth(rec_real_img_mask1) + self.compute_loss_smooth(rec_real_img_mask2)) / 2 * self.config.lambda_mask_smooth


        # combine losses
        return self.loss_g_masked_fake + self.loss_g_masked_cond + self.loss_g_cyc +self.loss_g_sm +\
               self.loss_g_mask_1 + self.loss_g_mask_2 + self.loss_g_mask_1_smooth + self.loss_g_mask_2_smooth

    def forward_D(self,real_img1,real_img2,desired_cond):  ##calculate the loss_D

        d_real_img_prob0, d_real_img_cond0 = self.D0.forward(self.real_img)
        d_real_img_prob1, d_real_img_cond1 = self.D1.forward(self.real_img)
        d_real_img_prob2, d_real_img_cond2 = self.D2.forward(self.real_img)

        self.loss_d_real = (self.compute_loss_D(d_real_img_prob0, True) +
                            self.compute_loss_D(d_real_img_prob1, True) +
                            self.compute_loss_D(d_real_img_prob2, True)) / 3 * self.config.lambda_D_prob
        self.loss_d_cond = (self.criterion_D_cond(d_real_img_cond0, self.real_cond) +
                            self.criterion_D_cond(d_real_img_cond1, self.real_cond) +
                            self.criterion_D_cond(d_real_img_cond2, self.real_cond)) / 3 * self.config.lambda_D_cond

        # generate fake images
        fake_imgs, fake_img_mask = self.G_A.forward(real_img1, desired_cond)
        fake_img_mask = self.do_if_necessary_saturate_mask(fake_img_mask, saturate=self.config.do_saturate_mask)
        fake_imgs_masked1 = fake_img_mask * self.real_img + (1 - fake_img_mask) * fake_imgs

        # D(fake_I)
        d_fake_desired_img_prob0, _ = self.D0.forward(fake_imgs_masked1.detach())
        d_fake_desired_img_prob1, _ = self.D1.forward(fake_imgs_masked1.detach())
        d_fake_desired_img_prob2, _ = self.D2.forward(fake_imgs_masked1.detach())

        fake_imgs2, fake_img_mask2 = self.G_B.forward(real_img2, desired_cond)
        fake_img_mask2 = self.do_if_necessary_saturate_mask(fake_img_mask2, saturate=self.config.do_saturate_mask)
        fake_imgs_masked2 = fake_img_mask2 * self.real_img + (1 - fake_img_mask2) * fake_imgs2

        d_fake_desired_img_prob2_0, _ = self.D0.forward(fake_imgs_masked2.detach())
        d_fake_desired_img_prob2_1, _ = self.D1.forward(fake_imgs_masked2.detach())
        d_fake_desired_img_prob2_2, _ = self.D2.forward(fake_imgs_masked2.detach())

        self.loss_d_fake = (self.compute_loss_D(d_fake_desired_img_prob0, False) +
                            self.compute_loss_D(d_fake_desired_img_prob1, False) +
                            self.compute_loss_D(d_fake_desired_img_prob2, False) +
                            self.compute_loss_D(d_fake_desired_img_prob2_0, False) +
                            self.compute_loss_D(d_fake_desired_img_prob2_1, False) +
                            self.compute_loss_D(d_fake_desired_img_prob2_2, False)) / 6 * self.config.lambda_D_prob

        # combine losses
        return self.loss_d_real + self.loss_d_cond + self.loss_d_fake,  fake_imgs_masked1.detach(),  fake_imgs_masked2.detach()

    def gradinet_penalty_D(self, model, fake_imgs_masked):
        # interpolate sample
        alpha = torch.rand(self.B, 1, 1, 1).cuda().expand_as(self.real_img)
        interpolated = Variable(alpha * self.real_img.data + (1 - alpha) * fake_imgs_masked.data, requires_grad=True)
        interpolated_prob, _ = model(interpolated)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        if model._name == 'D0':
            self.loss_d_gp0 = torch.mean((grad_l2norm - 1) ** 2) * self.config.lambda_D_gp
            return self.loss_d_gp0
        elif model._name == 'D1':
            self.loss_d_gp1 = torch.mean((grad_l2norm - 1) ** 2) * self.config.lambda_D_gp
            return self.loss_d_gp1
        else:
            self.loss_d_gp2 = torch.mean((grad_l2norm - 1) ** 2) * self.config.lambda_D_gp
            return self.loss_d_gp2

    def compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)

    def compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([
            # ('g_fake', self.loss_g_fake.item()),
            # ('g_cond', self.loss_g_cond.item()),
            ('g_mskd_fake', self.loss_g_masked_fake.item()),
            ('g_mskd_cond', self.loss_g_masked_cond.item()),
            # ('g_mskd_age', self.loss_g_masked_age.item()),
            # ('g_mskd_gender', self.loss_g_masked_gender.item()),
            ('g_cyc', self.loss_g_cyc.item()),
            # ('g_idt', self.loss_g_idt.item()),
            # ('g_pix', self.loss_g_pix.item()),
            # ('g_rgb_s', self.loss_g_fake_imgs_smooth.item()),
            ('g_m1', self.loss_g_mask_1.item()),
            ('g_m2', self.loss_g_mask_2.item()),
            ('g_m1_s', self.loss_g_mask_1_smooth.item()),
            ('g_m2_s', self.loss_g_mask_2_smooth.item()),
            # ('g_sm', self.loss_g_sm.item()),
            ('d_real', self.loss_d_real.item()),
            ('d_cond', self.loss_d_cond.item()),
            # ('d_age', self.loss_d_cond_age.item()),
            # ('d_gender', self.loss_d_cond_gender.item()),
            ('d_fake', self.loss_d_fake.item()),
            ('d_gp', self.loss_d_gp.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self.current_lr_G), ('lr_D', self.current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # input visuals
        title_input_img = os.path.basename(self.input_real_img_path[0])
        visuals['1_input_img'] = plot_utils.plot_au(self.vis_real_img, self.vis_real_cond, title=title_input_img)
        visuals['2_fake_img'] = plot_utils.plot_au(self.vis_fake_img, self.vis_desired_cond)
        visuals['3_rec_real_img'] = plot_utils.plot_au(self.vis_rec_real_img, self.vis_real_cond)
        visuals['4_fake_img_unmasked'] = self.vis_fake_img_unmasked
        visuals['5_fake_img_mask'] = self.vis_fake_img_mask
        visuals['6_rec_real_img_mask'] = self.vis_rec_real_img_mask
        visuals['7_cyc_img_unmasked'] = self.vis_fake_img_unmasked
        # visuals['8_fake_img_mask_sat'] = self.vis_fake_img_mask_saturated
        # visuals['9_rec_real_img_mask_sat'] = self.vis_rec_real_img_mask_saturated
        visuals['10_batch_real_img'] = self.vis_batch_real_img
        visuals['11_batch_fake_img'] = self.vis_batch_fake_img
        visuals['12_batch_fake_img_mask'] = self.vis_batch_fake_img_mask
        # visuals['11_idt_img'] = self.vis_idt_img

        return visuals

    def save(self, label):
        # save networks
        if label < 10:
            label = '00' + str(label)
        elif label < 100:
            label = '0' + str(label)

        self.save_network(self.G_A, 'G_A', label)
        # self.save_network(self.G_B, 'G_B', label)
        #
        # self.save_network(self.D0, 'D0', label)
        # self.save_network(self.D1, 'D1', label)
        # self.save_network(self.D2, 'D2', label)
        #
        # # save optimizers
        # self.save_optimizer(self.optimizer_G_A, 'G_A', label)
        # self.save_optimizer(self.optimizer_G_B, 'G_B', label)
        # self.save_optimizer(self.optimizer_D0, 'D0', label)
        # self.save_optimizer(self.optimizer_D1, 'D1', label)
        # self.save_optimizer(self.optimizer_D2, 'D2', label)

        # self.clean_checkpoint(self.save_dir)
    #
    def clean_checkpoint(self, checkpoint_dir, keep_num=4):
        if keep_num > 0:
            # names = list(sorted(glob.glob(os.path.join(checkpoint_dir, 'opt_*.pth'))))
            # if len(names) > keep_num:
            #     for name in names[:-keep_num * 2]:
            #         print(f"Deleting obslete checkpoint file {name}")
            #         os.remove(name)

            names = list(sorted(glob.glob(os.path.join(checkpoint_dir, 'net_*.pth'))))
            if len(names) > keep_num:
                for name in names[:-keep_num * 5]:
                    print(f"Deleting obslete checkpoint file {name}")
                    os.remove(name)

    def load(self):
        load_epoch = self.config.load_epoch

        # load G
        self.load_network(self.G_A, 'G_A', load_epoch)
        self.load_network(self.G_B, 'G_B', load_epoch)

        if self.is_train:
            # load D
            self.load_network(self.D0, 'D0', load_epoch)
            self.load_network(self.D1, 'D1', load_epoch)
            self.load_network(self.D2, 'D2', load_epoch)

            # # load optimizers
            # self.load_optimizer(self.optimizer_G_A, 'G_A', load_epoch)
            # self.load_optimizer(self.optimizer_G_B, 'G_B', load_epoch)
            # self.load_optimizer(self.optimizer_D0, 'D0', load_epoch)
            # self.load_optimizer(self.optimizer_D1, 'D1', load_epoch)
            # self.load_optimizer(self.optimizer_D2, 'D2', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G
        lr_decay_G = self.config.lr_G / self.config.nepochs_decay
        self.current_lr_G -= lr_decay_G
        for param_group in self.optimizer_G_A.param_groups:
            param_group['lr'] = self.current_lr_G
        for param_group in self.optimizer_G_B.param_groups:
            param_group['lr'] = self.current_lr_G
        print('update G learning rate: %f -> %f' % (self.current_lr_G + lr_decay_G, self.current_lr_G))

        # update learning rate D
        lr_decay_D = self.config.lr_D / self.config.nepochs_decay
        self.current_lr_D -= lr_decay_D
        for param_group in self.optimizer_D0.param_groups:
            param_group['lr'] = self.current_lr_D
        for param_group in self.optimizer_D1.param_groups:
            param_group['lr'] = self.current_lr_D
        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = self.current_lr_D
        print('update D learning rate: %f -> %f' % (self.current_lr_D + lr_decay_D, self.current_lr_D))

    def l1_loss_with_target_gradients(self, input, target):
        return torch.sum(torch.abs(input - target)) / input.data.nelement()

    def do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55 * torch.tanh(3 * (m - 0.5)) + 0.5, 0, 1) if saturate else m
