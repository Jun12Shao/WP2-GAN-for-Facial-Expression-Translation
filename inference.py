import os
import cv2
from utils import cv_utils
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from models.aegan import AE_GAN
from options import Options
from tqdm import tqdm
import pickle


class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = Options().parse()
        self._opt.mode='test'
        self._model = AE_GAN(self._opt)
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
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['fake_imgs_masked'],imgs['concat']

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)

        file_dir=filename[:-len(os.path.basename(filename))]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)


def main():
    opt =Options().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    with open(opt.test_root+opt.test_namelist, 'rb') as f:
        test_namelist= pickle.load(f, encoding='utf-8')
        f.close()

    with open(opt.test_root + opt.test_attr, 'rb') as f:
        test_dict = pickle.load(f, encoding='utf-8')
        f.close()

    input_path = 'D:/Datasets/CACD'

    morph = MorphFacesInTheWild(opt)
    new_dict={}

    i=0
    new_dict={}
    for id in tqdm(test_namelist):
        name=os.path.basename(id)
        i+=1
        real_cond=np.zeros(7)
        real_cond[test_dict[id]]=1

        filepath = input_path + id + '.jpg'

        for ex in range(7):
            dest_cond=np.zeros(7)
            dest_cond[ex]=1

            gen_img, _ = morph.morph_file(filepath, dest_cond, real_cond)

            output_name =name+'_'+str(ex) + '.jpg'

            output_path= opt.output_dir +'/'+ output_name
            new_id = name + '_' + str(ex)
            new_dict[new_id]=ex
            morph._save_img(gen_img, output_path)


    with open(opt.output_dir + '/'+'attr_for_generation.pkl', 'wb') as f:
        pickle.dump(new_dict, f)
        f.close()


if __name__ == '__main__':
    main()