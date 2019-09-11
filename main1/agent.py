#NATIVE LIBRARY IMPORTS
import os
import random
import math
from collections import namedtuple

#OPEN SOURCE IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F

#CUSTOM IMPORTS
import resnet

#######################################################################################################
#######################################################################################################

#REPLAY MEMORY FOR TRAINING OR MODEL SELECTION NETWORK
Transition = namedtuple('Transition',('state', 'action', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity,device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device=device

    def push(self,state,action,reward,done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        state = state.cpu().squeeze(0)
        action = torch.Tensor([action]).long()
        reward = torch.Tensor([reward])
        done = torch.Tensor([done])

        self.memory[self.position] = Transition(state,action,reward,done)
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        s = [None] * batch_size
        a = [None] * batch_size
        r = [None] * batch_size
        for i, e in enumerate(experiences):
            s[i] = e.state
            a[i] = e.action
            r[i] = e.reward

        s = torch.stack(s).to(self.device)
        a = torch.stack(a).to(self.device)
        r = torch.stack(r).to(self.device)

        return s,a,r

    def __len__(self):
        return len(self.memory)
#######################################################################################################

#OUR ENCODER DECODER NETWORK FOR MODEL SELECTION NETWORK
class Model(nn.Module):
    def __init__(self,action_space=10):
        super(Model,self).__init__()

        self.encoder = resnet.resnet18()
        self.decoder = torch.nn.Sequential(
                    nn.Linear(1000,512),
                    nn.ReLU(),
                    nn.Linear(512,256),
                    nn.ReLU(),
                    nn.Linear(256,action_space)
                )

    def encode(self,x):
        return torch.tanh(self.encoder(x))

    def decode(self,x):
        return self.decoder(x)

    def forward(self,x):
        latent_vector = self.encode(x)
        return self.decode(latent_vector)

#######################################################################################################

#AGENT COMPRISES OF A MODEL SELECTION NETWORK AND MAKES ACTIONS BASED ON IMG PATCHES
class Agent():
    def __init__(self,args):

        #RANDOM MODEL INITIALIZATION FUNCTION
        def init_weights(m):
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        #INITIALIZE HYPER PARAMS
        self.steps = 0
        self.BATCH_SIZE = args.batch_size
        self.GAMMA = args.gamma
        self.EPS_START = args.eps_start
        self.EPS_END = args.eps_end
        self.EPS_DECAY = args.eps_decay
        self.TARGET_UPDATE = args.target_update
        self.ACTION_SPACE = args.action_space
        self.device = args.device
        self.memory = ReplayMemory(args.memory_size,device=self.device)
        load = args.loadagent

        #INITIALIZE THE MODELS
        #self.model_target = Model()
        self.model = Model(action_space=self.ACTION_SPACE)
        if load:
            chkpoint = torch.load("model/agent.pth")
            #self.model_target.load_state_dict(chkpoint['model'])
            self.model.load_state_dict(chkpoint['model'])
        else:
            self.model.apply(init_weights)
            #self.model_target.apply(init_weights)
        #self.model_target.to(self.device)
        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.00001)

        return None

    #GIVEN STATE s GET ACTION a
    def selectAction(self,s):
        self.model.eval()
        with torch.no_grad():
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
            self.steps += 1
            if sample > eps_threshold:
                return [self.model(s).max(1)[1].item()]   #GREEDY ACTION
            else:
                n = s.shape[0]
                return [random.randrange(self.ACTION_SPACE) for i in range(n)]      #RANDOM ACTION

    #GREEDY ACTION SELECTION METHOD
    def greedyAction(self,s):
        self.model.eval()
        with torch.no_grad():
            return self.model(s).max(1)[1].item()   #GREEDY ACTION

    #OUR OPTIMIZATION FUNCTION
    def optimize(self):

        if len(self.memory) < self.BATCH_SIZE: return 0.0
        self.model.train()
        s,a,r = self.memory.sample(self.BATCH_SIZE)

        #predictions should be the PSNR SCORE
        qvals = self.model(s)
        qvals = qvals.gather(1,a)

        #CALCULATE LOSS
        self.opt.zero_grad()
        loss = F.mse_loss(qvals,r)
        loss.backward()
        self.opt.step()

        return loss.item()

    #OUR LEARNING FUNCTION
    def learn(self,s,a,psnr,ssim):

        self.memory.push(s,a,psnr,True)
        return self.optimize()

#######################################################################################################
#######################################################################################################

#SOME TESTING CODE TO MAKE SURE THIS FILE WORKS
if __name__ == "__main__":

    device = 'cuda'
    m = Model()
    m.to(device)
    img = torch.rand((3,100,100)).unsqueeze(0).to(device)

    print('Img shape: ', img.shape)
    print('out shape: ', m(img).shape)
    print('encode shape: ', m.encode(img).shape)
    print('decode shape: ', m.decode(m.encode(img)).shape)


