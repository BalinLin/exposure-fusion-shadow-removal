#!/usr/bin/python3
# output for mask_threshold

import argparse
import sys
import os
import cv2
from skimage.filters import threshold_otsu
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=400, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()

### ISTD
opt.dataroot_A = '/home/balin/exper/shadow_removal/Auto-Exposure/data/ISTD_Dataset/test/test_A'
opt.dataroot_B = '/home/balin/exper/shadow_removal/Auto-Exposure/data/ISTD_Dataset/test/test_C'

opt.im_suf_A = '.png'
opt.im_suf_B = '.png'

### SRD
# opt.dataroot_A = '/home/balin/exper/shadow_removal/Auto-Exposure/data/SRD_Dataset/test_data/shadow'
# opt.dataroot_B = '/home/balin/exper/shadow_removal/Auto-Exposure/data/SRD_Dataset/test_data/shadow_free'
#
# opt.im_suf_A = '.jpg'
# opt.im_suf_B = '.jpg'

### USR
# opt.dataroot_A = '/home/xwhu/dataset/shadow_USR/shadow_test'
# opt.dataroot_B = '/home/xwhu/dataset/shadow_USR/shadow_free'
# 
# opt.im_suf_A = '.jpg'
# opt.im_suf_B = '.jpg'

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')

print(opt)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

# Dataset loader
img_transform = transforms.Compose([
    transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)

###### Testing######
# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')
if not os.path.exists('output/mask'):
    os.makedirs('output/mask')
# if not os.path.exists('output/recovered_shadow'):
#     os.makedirs('output/recovered_shadow')
# if not os.path.exists('output/same_A'):
#     os.makedirs('output/same_A')
# if not os.path.exists('output/recovered_shadow_free'):
#     os.makedirs('output/recovered_shadow_free')
# if not os.path.exists('output/same_B'):
#     os.makedirs('output/same_B')

# ref code for Otsu
def mask_generator_output(shadow, shadow_free):
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    return mask

##################################### A to B // shadow to shadow-free
gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

for idx, img_name in enumerate(gt_list):
    # image = cv2.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A))
    # thresh = threshold_otsu(image)   #返回一个阈值
    # print("idx,thresh",idx, thresh)
    # dst = (image <= thresh)*1.0   #根据阈值进行分割

    # plt.figure('thresh',figsize=(8,8))

    # plt.subplot(121)
    # plt.title('original image')
    # plt.imshow(image,plt.cm.gray)

    # plt.subplot(122)
    # plt.title('binary image')
    # plt.imshow(dst,plt.cm.gray)

    # plt.show()

    print('predicting: %d / %d' % (idx + 1, len(gt_list)))

    # Set model input
    img = Image.open(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)).convert('RGB')
    img2 = Image.open(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B)).convert('RGB')

    shadow = (img_transform(img).unsqueeze(0)).to(device)
    shadow_free = (img_transform(img2).unsqueeze(0)).to(device)

    ## generator
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))

    # diff [0 - 255], mask[-1 or 1]
    diff = (np.asarray(im_f, dtype='float32') - np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow

    # im_s.save('output/A/%s' % img_name + opt.im_suf_A)
    # im_f.save('output/B/%s' % img_name + opt.im_suf_B)

    # maskImg = to_pil(mask.data.squeeze(0).cpu())
    # maskImg = maskImg.filter(ImageFilter.MedianFilter)
    # maskImg.save('output/mask/%s' % img_name + opt.im_suf_B)
    
    print('Generated images %04d of %04d' % (idx+1, len(gt_list)))