import numpy as np
import argparse
import torch
from glob import glob
from ntpath import basename
from scipy.misc import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

psnr = []
ssim = []
mae = []
names = []

files = list(glob(path_true + '/*/*.jpg')) + list(glob(path_true + '/*/*.png')) + list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))

for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)
    
    img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)
    
    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, multichannel=True))
    mae.append(compare_mae(img_gt, img_pred))

print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
)
