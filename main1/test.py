#NATIVE IMPORTS
import os
import glob
import argparse

#OPEN SOURCE IMPORTS
import cv2
import numpy as np
import torch

#CUSTOM IMPORTS
import RRDBNet_arch as arch
from utils import util

####################################################################################################
####################################################################################################
####################################################################################################

class Tester():

    def __init__(self,args):

        self.modelpath = args.srmodel_path
        self.upsize = args.upsize
        self.device = args.device

        #CURRENT MODE OF OP
        if args.mode == 'RRDB':
            self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            self.model.load_state_dict(torch.load(srmodel_path), strict=True)
            self.model = self.model.to(args.device)

    #evaluate image
    def eval(x):
        for path in glob.glob(test_img_folder):
            idx += 1
            base = os.path.splitext(os.path.basename(path))[0]

            #read images
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            #
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

        return self.model(x)

####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default="RRDB",help='Mode of Operation')
    parser.add_argument("--srmodel_path",default="../models/RRDB_ESRGAN_x4.pth", help='Path to the SR model')
    parser.add_argument("--action_space",default=4, help='Action Space size')
    parser.add_argument("--down_method",default="BI",help='method of downsampling. [BI|BD]')
    parser.add_argument("--upsize", default=4,help="Upsampling size of the network")
    parser.add_argument("--device",default='cuda:0',help='set device to train on')
    args = parser.parse_args()

####################################################################################################
####################################################################################################
####################################################################################################

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
#device = torch.device('cpu')

test_img_folder = os.path.join(args.file,'*')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model = model.to(device)
print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    print(img_LR.shape)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)



