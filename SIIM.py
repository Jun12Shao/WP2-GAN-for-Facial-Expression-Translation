import os
import cv2
from tqdm import tqdm
import pickle
import numpy as np

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def main():
    output_dir="'E:\GAN_test9/unet-cfee-bs-6-14/7-siim-sample"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open('D:\Datasets/CACD/2020-12-8/CFEE-image-pair.pkl', 'rb') as f:
        new_dict = pickle.load(f, encoding='utf-8')
        f.close()

    dirs=[
          'E:\GAN_test8\##GMi-pg-cfee-comp\#1-exp-cls(40)-fn',
        #   'E:\GAN_test8\##GMi-bs-cfee-comp\#1-exp-cls(95)-fn'
          ]
    for root1 in dirs:
        root2='D:\Datasets/CACD'
        SSIM_score=0

        ##################calculate SSIM
        n=0
        k=0
        for m in tqdm(new_dict):
            name1= root1+m+ '.jpg'
            name2= root2 + new_dict[m] + '.jpg'
            if os.path.exists(name1) and os.path.exists(name2):
                img1= cv2.imread(name1)
                img2= cv2.imread(name2)

                ssim=calculate_ssim(img1, img2)
                k+=1
                SSIM_score+=ssim

        print("SSIM score:", SSIM_score/k,k)


if __name__ == '__main__':
    main()