import time
from data_loader import get_loader
from models.aegan import AE_GAN
from options import Options
import pickle
import numpy as np
import torchvision.transforms as transforms
from utils import cv_utils
from PIL import Image
import cv2
import random
from fid_score import get_fid
from resnet import resnet18
from data_loader3 import get_loader as get_loader3
from torch.backends import cudnn
import torch
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class MorphFacesInTheWild:
    def __init__(self, opt, model):
        self._opt = opt
        self._model = model
        self._model.set_eval()
        self._transform = transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expresion, real_cond):
        face = cv_utils.read_cv2_img(img_path)

        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion), 0)
        real_cond = torch.unsqueeze(torch.from_numpy(real_cond), 0)

        test_batch = {'real_img': face, 'real_cond': real_cond, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}   ##此处real_condition 有问题，应该是新输入的expression
        self._model.set_input(test_batch)
        imgs = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['fake_imgs_masked']

    def _save_img(self, img, filename):
        filepath = filename

        file_dir=filename[:-len(os.path.basename(filename))]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

def evaluate(model, data_loader):
    """
    Calculate classification error (%) for given model
    and data set.

    Parameters:

    - model: A Trained Pytorch Model
    - data_loader: A Pytorch data loader object
    """

    y_true = np.array([], dtype=np.int)
    y_pred = np.array([], dtype=np.int)

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # outputs = model(inputs)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true = np.concatenate((y_true, labels.cpu()))
            y_pred = np.concatenate((y_pred, predicted.cpu()))

    error = np.sum(y_pred != y_true) / len(y_true)
    return error

def test(opt, model, i_epoch):
    # model.set_eval()

    if not os.path.isdir(opt.buffer_dir):
        os.makedirs(opt.buffer_dir)

    # with open(opt.aus_root + '/' + opt.aus_file2, 'rb') as f:
    #     target_dict2 = pickle.load(f, encoding='utf-8')
    #     f.close()
    with open(opt.aus_root + '/' + opt.aus_file, 'rb') as f:
        target_dict = pickle.load(f, encoding='utf-8')
        f.close()

    with open(opt.aus_root + '/' + opt.label_index_file, 'rb') as f:
        index_dict = pickle.load(f, encoding='utf-8')
        f.close()

    with open(opt.aus_root + '/' + opt.aus_test_ids_file, 'rb') as f:
        test_namelist = pickle.load(f, encoding='utf-8')
        f.close()

    #########################################################################################

    morph = MorphFacesInTheWild(opt, model)
    pres = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
    name_label = {}
    for id in test_namelist[:1200]:
        name = os.path.basename(id)

        real_cond = target_dict[id][:17] / 5.0

        for pre in pres:
            filepath = opt.aus_root + id + pre
            if os.path.exists(filepath):

                for i in range(7):
                    for j in range(2):
                        sample_nm = random.choice(index_dict[i])
                        des_cond = target_dict[sample_nm] / 5.0

                        gen_img = morph.morph_file(filepath, des_cond, real_cond)
                        filename = name + '_' + str(i)+ '_' + str(j) + '.jpg'
                        output_name = os.path.join(opt.buffer_dir, filename)

                        new_id = name + '_' + str(i)+ '_' + str(j)
                        name_label[new_id] = i

                        morph._save_img(gen_img, output_name)
                break

    with open(opt.buffer_dir + 'des_labels_for_generation.pkl', 'wb') as f:
        pickle.dump(name_label, f)
        f.close()

    ## evaluate the result
    cudnn.benchmark = True

    # Data loader.
    test_set = get_loader3(config, 'CR', 'test', config.num_workers)

    # model = ResNet(n, shortcuts=False)
    model = resnet18()

    model_file = os.path.join(config.checkpoints_dir, 'cfee_resnet18_g7-6-13.pt')



    # Record metrics
    model.cuda()
    device = torch.device("cuda")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    test_err = evaluate(model, test_set)

    print(f'test_acc: {1 - test_err}')

    return test_err

def sample(opt,model,data_loader,i_epoch):
    model.set_eval()

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    with open(opt.aus_root +'/'+ opt.aus_file, 'rb') as f:
        target_dict = pickle.load(f, encoding='utf-8')
        f.close()

    with open(opt.aus_root + '/' + opt.label_index_file, 'rb') as f:
        index_dict = pickle.load(f, encoding='utf-8')
        f.close()

    for i_val_batch, val_batch in enumerate(data_loader):
        break

    img_ext = None
    for j in range(7):
        sample_nm=random.choice(index_dict[j])
        cond=target_dict[sample_nm]/5.0

        expression = torch.unsqueeze(torch.from_numpy(cond), 0)
        expression = expression.expand(15, -1)
        val_batch['desired_cond'] = expression
        model.set_input(val_batch)
        imgs = model.forward(keep_data_for_visuals=False, return_estimates=True)

        img = imgs['fake_imgs_masked_batch']

        if img_ext is None:
            img_ext = imgs['real_img_batch']
            img_ext = np.concatenate([img_ext, img], 1)
        else:
            img_ext = np.concatenate([img_ext, img], 1)

    filename = str(i_epoch) + '.jpg'
    filepath = os.path.join(opt.output_dir, filename)

    img = cv2.cvtColor(img_ext, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img)

    model.set_train()



def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    aus_loader_train = get_loader(config, 'AUs', 'train', config.num_workers)
    aus_loader_test = get_loader(config, 'AUs', 'test', config.num_workers)
    aus_loader_valid = get_loader(config, 'AUs', 'valid', config.num_workers)


    ## establish model.
    model = AE_GAN(config)
    print("Model %s was created" % model.name)

    train(config, model, aus_loader_train, aus_loader_test, aus_loader_valid)



def train(config, model, aus_loader_train, aus_loader_test,aus_loader_valid):
    dataset_train = aus_loader_train
    dataset_test = aus_loader_test

    dataset_train_size = len(dataset_train)
    dataset_test_size = len(dataset_test)
    print('#train images = %d' % dataset_train_size)
    print('#test images = %d' % dataset_test_size)

    total_steps = config.load_epoch * dataset_train_size

    # tb_visualizer = TBVisualizer(config)
    loss_errors = []
    mini_test_error=1
    for i_epoch in range(config.load_epoch + 1, config.nepochs_no_decay + config.nepochs_decay + 1):
        epoch_start_time = time.time()

        # train epoch
        epoch_iter = 0
        model.set_train()  ## set in training model.
        for i_train_batch, train_batch in enumerate(dataset_train):
            # train model
            model.set_input(train_batch)
            train_generator = ((i_train_batch + 1) % config.train_G_every_n_iterations == 0)
            model.optimize_parameters(train_generator=train_generator)

            # update epoch info
            total_steps += config.batch_size
            epoch_iter += config.batch_size

            ## record the loss values of both G and D
            if train_generator:
                errors = model.get_current_errors()
                loss_errors.append(errors)


        test_err=test(config, model, i_epoch )
        sample(config, model, aus_loader_valid, i_epoch)
        # print('saving the model at the end of epoch %d, iters %d' % (i_epoch, total_steps))
        # model.save(i_epoch)

        if  mini_test_error>test_err and 1-test_err>0.65:
            mini_test_error=test_err
            print('saving the model at the end of epoch %d, iters %d' % (i_epoch, total_steps))
            model.save(i_epoch)
        #     sample(config, model, aus_loader_valid, i_epoch)
        #     get_fid(config.buffer_dir)

        # elif 1-test_err>0.68:
        #     print('saving the model at the end of epoch %d, iters %d' % (i_epoch, total_steps))
        #     model.save(i_epoch)
        #     sample(config, model, aus_loader_valid, i_epoch)
        #     get_fid(config.buffer_dir)
            # sample(config, model, aus_loader_valid, i_epoch)

        model.set_train()

        # print epoch info
        time_epoch = time.time() - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
              (i_epoch, config.nepochs_no_decay + config.nepochs_decay, time_epoch,
               time_epoch / 60, time_epoch / 3600))

        # update learning rate
        if i_epoch > config.nepochs_no_decay:
            model.update_learning_rate()

    # print('test_acc', 1-mini_test_error)

if __name__ == '__main__':
    config = Options().parse()
    print(config)
    main(config)
