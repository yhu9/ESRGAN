#NATIVE IMPORTS
import os
import glob
import argparse
from collections import deque
from itertools import count
import random
import time

#OPEN SOURCE IMPORTS
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction import image
from skimage.util.shape import view_as_windows

#CUSTOM IMPORTS
import RRDBNet_arch as arch
import agent
import logger
from utils import util

########################################################################################################
#ARGUMENTS TO PASS FOR TRAINING
parser = argparse.ArgumentParser()
parser.add_argument("--srmodel_path",default="../models/RRDB_ESRGAN_x4.pth", help='Path to the SR model')
parser.add_argument("--batch_size",default=32, help='Batch Size')
parser.add_argument("--gamma",default=.9, help='Gamma Value for RL algorithm')
parser.add_argument("--eps_start",default=.90, help='Epsilon decay start value')
parser.add_argument("--eps_end",default=0.10, help='Epsilon decay end value')
parser.add_argument("--eps_decay",default=10000, help='Epsilon decay fractional step size')
parser.add_argument("--target_update",default=20, help='Target network update time')
parser.add_argument("--action_space",default=4, help='Action Space size')
parser.add_argument("--memory_size",default=10000, help='Memory Size')
parser.add_argument("--training_lrpath",default="../../../data/DIV2K_train_LR_bicubic/X4")
#parser.add_argument("--training_lrpath",default="LR")
parser.add_argument("--training_hrpath",default="../../../data/DIV2K_train_HR")
parser.add_argument("--testing_path",default="../../../data/DIV2K_train_LR_bicubic/X4")
parser.add_argument("--loadagent",default=False, action='store_const',const=True)
parser.add_argument("--learning_rate",default=0.0001,help="Learning rate of Super Resolution Models")
parser.add_argument("--upsize", default=4,help="Upsampling size of the network")
parser.add_argument("--gen_patchinfo",default=False,action='store_const',const=True)
parser.add_argument("--name", required=True, help='Name to give this training session')
args = parser.parse_args()
########################################################################################################

#OUR END-TO-END MODEL TRAINER
class SISR():
    def __init__(self, args=args):

        #RANDOM MODEL INITIALIZATION FUNCTION
        def init_weights(m):
            if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        #INITIALIZE VARIABLES
        self.SR_COUNT = args.action_space
        SRMODEL_PATH = args.srmodel_path
        self.batch_size = args.batch_size
        self.TRAINING_LRPATH = glob.glob(os.path.join(args.training_lrpath,"*"))
        self.TRAINING_HRPATH = glob.glob(os.path.join(args.training_hrpath,"*"))
        self.TRAINING_LRPATH.sort()
        self.TRAINING_HRPATH.sort()
        self.TESTING_PATH = glob.glob(os.path.join(args.testing_path,"*"))
        self.LR = args.learning_rate
        self.UPSIZE = args.upsize
        self.step = 0
        if args.name != 'none':
            self.logger = logger.Logger(args.name)   #create our logger for tensorboard in log directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #determine cpu/gpu

        #LOAD A COPY OF THE MODEL N TIMES
        self.SRmodels = []
        self.SRoptimizers = []
        for i in range(self.SR_COUNT):
            model = arch.RRDBNet(3,3,64,23,gc=32)
            model.apply(init_weights)
            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)
            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=self.LR))
        print('Model path {:s}. Loaded...'.format(SRMODEL_PATH))

    #TRAINING IMG LOADER WITH VARIABLE PATCH SIZES AND UPSCALE FACTOR
    def getTrainingPatches(self,LR,HR, patch_size=16,stride=16):

        #ENSURE BOXES of size PATCH_SIZE CAN FIT OVER ENTIRE IMAGE
        h,w,d = LR.shape
        padh = patch_size - (h % patch_size)
        padw = patch_size - (w % patch_size)
        h = h + padh
        w = w + padw
        lrh, lrw = h,w
        LR = np.pad(LR,pad_width=((0,padh),(0,padw),(0,0)), mode='symmetric')       #symmetric padding to allow meaningful edges
        h,w,d = HR.shape
        padh = (patch_size*self.UPSIZE) - (h % (patch_size*self.UPSIZE))
        padw = (patch_size*self.UPSIZE) - (w % (patch_size*self.UPSIZE))
        h = h + padh
        w = w + padw
        hrh,hrw = h,w
        HR = np.pad(HR,pad_width=((0,padh),(0,padw),(0,0)),mode='symmetric')

        #GET PATCHES USING NUMPY'S VIEW AS WINDOW FUNCTION
        maxpatch_lr = (lrh // patch_size) * (lrw // patch_size)
        maxpatch_hr = (hrh // (patch_size*self.UPSIZE)) * (hrw // (patch_size*self.UPSIZE))
        LRpatches = view_as_windows(LR,(patch_size,patch_size,3),stride)
        HRpatches = view_as_windows(HR,(patch_size*self.UPSIZE,patch_size*self.UPSIZE,3),stride*4)

        #RESHAPE CORRECTLY AND CONVERT TO PYTORCH TENSOR
        LRpatches = torch.from_numpy(LRpatches).float()
        HRpatches = torch.from_numpy(HRpatches).float()
        LRpatches = LRpatches.permute(2,0,1,3,4,5).contiguous().view(-1,patch_size,patch_size,3)
        HRpatches = HRpatches.permute(2,0,1,3,4,5).contiguous().view(-1,patch_size*self.UPSIZE,patch_size*self.UPSIZE,3)
        LRpatches = LRpatches.permute(0,3,1,2)
        HRpatches = HRpatches.permute(0,3,1,2)
        LRpatches = LRpatches * 1.0 / 255
        HRpatches = HRpatches * 1.0 / 255

        return LRpatches.to(self.device),HRpatches.to(self.device)

    #APPLY SISR on a LR patch AND OPTIMIZE THAT PARTICULAR SISR MODEL ON CORRESPONDING HR PATCH
    def applySISR(self,lr,action,hr):

        self.SRoptimizers[action].zero_grad()
        hr_hat = self.SRmodels[action](lr)
        loss = F.l1_loss(hr,hr_hat)
        loss.backward()
        self.SRoptimizers[action].step()

        hr_hat = hr_hat.squeeze(0).permute(1,2,0); hr = hr.squeeze(0).permute(1,2,0)
        hr_hat = hr_hat.detach().cpu().numpy()
        hr = hr.detach().cpu().numpy()
        psnr,ssim = util.calc_metrics(hr_hat,hr,crop_border=self.UPSIZE)

        return hr_hat, psnr, ssim, loss.item()

    #SAVE THE AGENT AND THE SISR MODELS INTO A SINGLE FILE
    def savemodels(self):
        data = {}
        data['agent'] = self.agent.model.state_dict()
        for i,m in enumerate(self.SRmodels):
            modelname = "sisr" + str(i)
            data[modelname] = m.state_dict()
        torch.save(data,"models/sisr.pth")

    #MAIN FUNCTION WHICH GETS PATCH INFO GIVEN CURRENT TRAINING SET AND PATCHSIZE
    def genPatchInfo(self):
        data = []
        for idx in range(len(self.TRAINING_HRPATH)):
            HRpath = self.TRAINING_HRPATH[idx]
            LRpath = self.TRAINING_LRPATH[idx]
            LR = cv2.imread(LRpath,cv2.IMREAD_COLOR)
            HR = cv2.imread(HRpath,cv2.IMREAD_COLOR)
            LR,HR = self.getTrainingPatches(LR,HR)
            data.append(LR.shape[0])
            print(LR.shape)
        np.save('models/patchinfo',np.array(data))
        print("num images", len(data))
        print("total patches", sum(data))

    #TRAINING REGIMEN
    def train(self):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #create our agent on based on previous information
        self.patchinfo = np.load('models/patchinfo.npy')
        self.agent = agent.Agent(args,self.device,args.action_space,self.patchinfo.sum())

        #START TRAINING
        for c in count():
            idx = random.randint(0,len(self.TRAINING_HRPATH) - 1)
            HRpath = self.TRAINING_HRPATH[idx]
            LRpath = self.TRAINING_LRPATH[idx]
            LR = cv2.imread(LRpath,cv2.IMREAD_COLOR)
            HR = cv2.imread(HRpath,cv2.IMREAD_COLOR)
            LR,HR = self.getTrainingPatches(LR,HR)
            sisr_loss = []
            agent_loss = []

            #FOR 10 RANDOM SUB SAMPLES
            for _ in range(10):
                labels = torch.Tensor(np.array(random.sample(range(len(LR)), 64))).long().cuda()
                lrbatch = LR[labels,:,:,:]
                hrbatch = HR[labels,:,:,:]

                self.agent.opt.zero_grad()    #zero our policy gradients
                #UPDATE OUR SISR MODELS
                for j,sisr in enumerate(self.SRmodels):
                    self.SRoptimizers[j].zero_grad()           #zero our sisr gradients
                    hr_pred = sisr(lrbatch)
                    m_labels = labels + np.sum(self.patchinfo[:idx])

                    #update sisr model based on weighted l1 loss
                    l1diff = torch.abs(hr_pred - hrbatch).view(64,-1).mean(1)
                    imgscore = torch.matmul(l1diff.unsqueeze(1),F.one_hot(torch.Tensor([j]).long(),self.SR_COUNT).float().to(self.device))
                    weighted_imgscore = self.agent.model(imgscore,m_labels)
                    loss1 = torch.mean(weighted_imgscore)
                    loss1.backward(retain_graph=True)
                    self.SRoptimizers[j].step()
                    sisr_loss.append(loss1.item())

                    #gather the gradients of the agent policy and constrain them to be within 0-1 with max value as 1
                    one_matrix = torch.ones(64,self.SR_COUNT).to(self.device)
                    weight_identity = self.agent.model(one_matrix,m_labels)
                    loss2 = torch.mean(torch.abs(torch.sum(torch.abs(weight_identity),dim=1) - 1)) #have sum of each row equal to 1
                    val,maxid = weight_identity.max(1) #have max of each row equal to 1
                    loss3 = torch.mean(torch.abs(weight_identity[:,maxid] - 1))
                    loss2.backward(retain_graph=True)
                    loss3.backward(retain_graph=True)
                    agent_loss.append(loss2.item() + loss3.item() + loss1.item())

                #UPDATE THE AGENT POLICY ACCORDING TO ACCUMULATED GRADIENTS FOR ALL SUPER RESOLUTION MODELS
                self.agent.opt.step()

                #LOG THE INFORMATION
                print('\rEpisode {}, Agent Loss: {:.4f}, SISR Loss: {:.4f}'\
                      .format(c,np.mean(agent_loss),np.mean(sisr_loss)),end="\n")
                self.logger.scalar_summary({'AgentLoss': np.mean(agent_loss), 'SISRLoss': np.mean(sisr_loss)})
                actions_taken = self.agent.model.M.weight.max(1)[1]
                self.logger.hist_summary('actions',actions_taken.cpu().numpy(),bins=self.SR_COUNT)

            #save the model at the end of every episode
            self.savemodels()

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    if args.gen_patchinfo:
        sisr.genPatchInfo()
    else:
        sisr.train()


