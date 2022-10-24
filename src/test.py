import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
import model
import numpy as np
from PIL import Image
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default='/LLIE/data/LOL/test/low')
parser.add_argument('--name', type=str, default='LOL')
parser.add_argument('--pretrain_path', type=str, default='checkpoint/Epoch89.pth')
parser.add_argument('--dim_hist', type=int, default=64)
args = parser.parse_args()
 
def lowlight(image_path, color_net):
    
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    
    with torch.no_grad():
        start = time.time()
        gray, color_hist, enhanced_image= color_net(data_lowlight)
    end_time = (time.time() - start)
    print(end_time)

    result_path = './results/' + args.name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    torchvision.utils.save_image(enhanced_image, result_path + image_path.split("/")[-1])

if __name__ == '__main__':
    
    with torch.no_grad():
        # model setting
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        color_net = model.color_net()
        color_net = nn.DataParallel(color_net)
        color_net = color_net.cuda()
        color_net.load_state_dict(torch.load(args.pretrain_path))

        # path setting
        filePath = args.filepath
        test_list = glob.glob(filePath + "/*")

        # inference
        for image in test_list:
            print(image)
            lowlight(image, color_net)
