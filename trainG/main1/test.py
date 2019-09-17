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
parser = argparse.ArgumentParser()
parser.add_argument("--mode",default="RRDB",help='Mode of Operation')
parser.add_argument("--srmodel_path",default="models/RRDB_ESRGAN_x4.pth", help='Path to the SR model')
parser.add_argument("--dataroot",default="../data/testing")
parser.add_argument("--action_space",default=4, help='Action Space size')
parser.add_argument("--down_method",default="BI",help='method of downsampling. [BI|BD]')
parser.add_argument("--upsize", default=4,help="Upsampling size of the network")
parser.add_argument("--device",default='cuda:0',help='set device to train on')
parser.add_argument("--name",default='none',help='set the name of this testing instance')
args = parser.parse_args()
####################################################################################################
####################################################################################################
####################################################################################################

class Tester():
    def __init__(self,args=args):
        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(args.srmodel_path), strict=True)
        self.model = self.model.to(args.device)
        self.device = args.device

        #test_img_folder = os.path.join(args.file,'*')
        downsample_method = args.down_method

        self.hr_rootdir = os.path.join(args.dataroot,'HR')
        self.lr_rootdir = os.path.join(args.dataroot,"LR" + downsample_method)
        self.validationsets = ['Set5','Set14','B100', 'Manga109','Urban100']
        #self.validationsets = ['Set5','Set14','B100']       #ESRGAN CANNOT TRAIN ON IMAGES LARGER THAN 96x96
        self.upsize = args.upsize
        self.resfolder = 'x' + str(args.upsize)

    #TEST THE ORIGINAL MODEL ON ALL DATASETS AND GATHER PSNR/SSIM SCORES
    def validate(self):
        scores = {}
        self.model.eval()
        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset)
            LR_dir = os.path.join(os.path.join(self.lr_rootdir,vset),self.resfolder)

            #APPLY MODEL ON LR IMAGES
            HR_files = [os.path.join(HR_dir, f) for f in os.listdir(HR_dir)]
            LR_files = [os.path.join(LR_dir, f) for f in os.listdir(LR_dir)]
            HR_files.sort()
            LR_files.sort()

            #PSNR/SSIM SCORE FOR CURRENT VALIDATION SET
            for hr_file, lr_file in zip(HR_files,LR_files):
                hr = cv2.imread(hr_file,cv2.IMREAD_COLOR)
                lr = cv2.imread(lr_file,cv2.IMREAD_COLOR) * 1.0 / 255
                lr_img = torch.from_numpy(np.transpose(lr[:,:,[2,1,0]],(2,0,1))).float()
                lr_img = lr_img.unsqueeze(0)
                lr_img = lr_img.to(self.device)

                out = self.model(lr_img).data.squeeze().float().cpu().clamp_(0,1).numpy()
                out = np.transpose(out[[2,1,0],:,:],(1,2,0))
                out = (out * 255.0).round()

                psnr,ssim = util.calc_metrics(hr,out,crop_border=self.upsize)
                scores[vset].append([psnr,ssim])

            mu_psnr = np.mean(np.array(scores[vset])[:,0])
            mu_ssim = np.mean(np.array(scores[vset])[:,1])
            print(vset + ' scores', mu_psnr,mu_ssim)

####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == '__main__':

    testing_regime = Tester()

    with torch.no_grad():
        testing_regime.validate()


    '''
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
    '''
