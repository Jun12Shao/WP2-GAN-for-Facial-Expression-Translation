
import torch.optim as optim
from options import Options
from resnet import resnet18
import numpy as np
# from data_loader_train import get_loader
from data_loader3 import get_loader
from torch.backends import cudnn
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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

def main(config):
    # For fast training.
    cudnn.benchmark = True

    for input_dir in ['E:\GAN_test8\##MD-pg-cfee-comp/1-exp-cls(35)']:
    # for input_dir in ['E:\GAN_test4\##MD-12-24-pg-rafd\#1-exp-cls(40)','C:/Users\sh_jun\PycharmProjects\Face\StarGAN_master_pytorch\stargan/results-RafD']:
    #,'E:\GAN_test5\##GMi-1-16-sp-rafd/1-exp-cls(102)' ,
        config.aus_root=input_dir


    # Data loader.
        aus_loader_train = get_loader(config, 'CR', 'train', config.num_workers)

        aus_loader_test = get_loader(config, 'CR', 'test', config.num_workers)

        # ## Train model
        # ns = [3, 5, 7, 9]
        # n = 3
        # lr = 0.1  # authors cite 0.1
        # momentum = 0.9
        # weight_decay = 0.0001
        # gamma = 0.1
        # epochs = 80
        # milestones = [82, 123]

        # model = resnet18()
        # train2(config, model, epochs, aus_loader_train, aus_loader_test, criterion, optimizer, results_file,
        #       scheduler=scheduler, MODEL_PATH=model_file)

        ## Test model
        model = resnet18()
        model_file = os.path.join(config.checkpoints_dir, 'CFEE_resnet18_g7-12-30_v2.pt')
        device = torch.device("cuda")
        model.load_state_dict(torch.load(model_file, map_location=device))

    # # Record metrics
    #     model.cuda()
    #     device = torch.device("cuda")
    #     model.load_state_dict(torch.load(model_file, map_location=device))

        model.eval()

        test_err = evaluate(model, aus_loader_test)

        print(f'test_acc: {1-test_err}')


if __name__ == '__main__':
    config = Options().parse()
    print(config)
    main(config)





