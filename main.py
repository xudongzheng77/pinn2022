import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import shutil
from train import train
import time
import math
from read_input_data import *

device=torch.device('cuda:0')
# torch.cuda.set_device(device)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cwd = os.getcwd()
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
mytype = torch.float

NMODE = 100
NSEC = 100
BATCH_SIZE = 20
LEARNING_RATE = 1e-2
MAX_EPOCHS = 100000 

midline = 22.31
flag_right = False

InputDir = 'inputs'
InputNMode = 100

### data size (for time)
data_index_start = 7400
data_index_end = 7800 
data_step = 20 

### data size (for shape)
num_shape_data = 5


### flow parameters
Psub = 10000.0
rho = 1.1e-3
alpha = 60.0
beta = 6.0e-05
i_ymin = 1 - 1   # index of nodal surface for the location of minimum y
i_ymax = 128 - 1 # index of nodal surface for the location of maximum y

### information of the points for interpolation in z direction
nInterp = NSEC
z0Interp = 22.42
z1Interp = 23.95
zmin = 22.77
zmax = 23.85
nz = nInterp

Psub = torch.Tensor([Psub])
rho = torch.Tensor([rho])
alpha = torch.Tensor([alpha])
beta = torch.Tensor([beta])

### delete output folder and create empty one
filePath = cwd+'/'+'outputs'
if os.path.exists(filePath):
    shutil.rmtree(filePath)
os.mkdir(filePath)

##############################################################
class lblock(nn.Module):
    def __init__(self,hidden_size):
        super(lblock, self).__init__()
        self.lb = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.ln(self.lb(x))+x

class Neural_Network1(nn.Module):
    def __init__(self,hiddensize=128):
        super(Neural_Network1, self).__init__()
        self.encoder = nn.LSTM(NSEC,hiddensize,1,batch_first=True)
        self.decoder = nn.LSTM(NSEC,hiddensize,1,batch_first=True)
        self.mlp = nn.Sequential(
            lblock(hiddensize),
            lblock(hiddensize),
            nn.Linear(hiddensize,NMODE),
        )
    
    def forward(self,x):
        _,(h,c) = self.encoder(x)
        tmp,_ = self.decoder(x,(h,c))
        return self.mlp(tmp)


class structtype():
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,lab,val):
        self.__dict__[lab] = val
	

class mydata(Dataset):
    def __init__(self):
        self.dt = tm[1]-tm[0]
        self.t = tm.unsqueeze(0).unsqueeze(2).repeat(1,1,1).to(device)
        self.x = xm.unsqueeze(0).to(device)
        self.normdx = (self.x-self.x.mean())/self.x.std()
        #self.rhs = rhs.to(device)
    def __getitem__(self, index):
        return self.normdx[index],self.t[index],self.x[index]#,self.rhs[index]
    def __len__(self):
        return self.normdx.shape[0]

	
def classtotorch(inclass):
    for attribute, _ in inclass.__dict__.items():
        obj = getattr(inclass, attribute)
        if isinstance(obj, np.ndarray):
            if obj.dtype == 'float64':
                obj=torch.tensor(obj, dtype=mytype, device=device)
                setattr(inclass, attribute, obj)
            else:
                obj=torch.tensor(obj, dtype=torch.int, device=device)
                setattr(inclass, attribute, obj)
		

def read_inputfiles(parameters):
    surf = read_surface_mesh(parameters.InputDir)
    vol = read_volume_mesh(parameters.InputDir)
    classtotorch(surf)
    classtotorch(vol)

    corres_s = read_corresponding_table(parameters.InputDir) - 1
    frequency, eigen_vx, eigen_vy, eigen_vz, C1, C2 = eigen(parameters.InputDir, vol.npt_v, parameters.InputNMode)
    omega = 2 * math.pi * frequency

    indm, tm_all, xm_all, zm_all = read_measurements(parameters.InputDir)

    # nmode: number of modes that be used in forming the 3D shape
    omega = omega[0:parameters.nmode]
    frequency = frequency[0:parameters.nmode]
    eigen_vx = eigen_vx[:, 0:parameters.nmode].to(device)
    eigen_vy = eigen_vy[:, 0:parameters.nmode].to(device)
    eigen_vz = eigen_vz[:, 0:parameters.nmode].to(device)

    ID = parameters.data_index_start+parameters.data_step*torch.linspace(0,parameters.batch_size-1,parameters.batch_size,dtype=torch.long)
    idx = torch.nonzero(torch.eq(indm.unsqueeze(1),ID),as_tuple=False)[:,0]

    tm = tm_all[idx].to(device)
    xm = xm_all[idx, :].to(device)
    zm = zm_all[idx, :].to(device)
    
    return surf, vol, corres_s, omega, frequency, eigen_vx, eigen_vy, eigen_vz, tm, xm, zm, C1, C2
	

def batchdata1(parameters):



    info = structtype(surf=surf, vol=vol, omega=omega, frequency=frequency, 
                      eigen_vx=eigen_vx, eigen_vy=eigen_vy, eigen_vz=eigen_vz,
                      tm=tm, xm=xm, zm=zm,
                      corres_s=corres_s, nsec=parameters.nsec, 
                      Psub=Psub, rho=rho, alpha=alpha, beta=beta,
                      )

    info.surf.xs = surf.xs.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)
    info.surf.ys = surf.ys.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)
    info.surf.zs = surf.zs.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)

    info.surf.xs0 = surf.xs0.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)
    info.surf.ys0 = surf.ys0.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)
    info.surf.zs0 = surf.zs0.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)

    info.eigen_vx = eigen_vx.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)
    info.eigen_vy = eigen_vy.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)
    info.eigen_vz = eigen_vz.repeat(parameters.batch_size, parameters.nsec, 1, 1).to(device)
	
    info.omega = omega.repeat(parameters.batch_size, 1).to(device)
    info.frequency = frequency.repeat(parameters.batch_size, 1).to(device)

    info.Psub = Psub.repeat([parameters.batch_size, 1]).to(device)
    info.rho = rho.repeat([parameters.batch_size, 1]).to(device)

    info.alpha = alpha.repeat([parameters.batch_size, 1]).to(device)
    info.beta = beta.repeat([parameters.batch_size, 1]).to(device)

    info.tm = tm.reshape(tm.size(0), 1).to(device)
    info.xm = xm
	
    return info

def find_included_indices():
    logical = ~surf.flag_exclude.eq(1)    
    logical = logical.unsqueeze(0).repeat([1, 1, surf.eles.size(1)-1]).reshape([logical.size(0), surf.eles.size(1)-1])
    E = surf.eles[:, 0:surf.eles.size(1)-1] - 1    
    e = E[logical].reshape(logical.size(0), 3)      
    k = e.unique()
    return k, e
	
	
if __name__=='__main__':


    print('')
    print('device = ', device)
    print('')

    NUM_BATCH = torch.floor(torch.tensor([torch.arange(data_index_start, data_index_end, data_step).size(0)])/BATCH_SIZE)
    NUM_BATCH = NUM_BATCH[0].to(dtype=torch.int64)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parameters = initialization(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, num_batch=NUM_BATCH, 
                    learning_rate=LEARNING_RATE, InputNMode=InputNMode, nmode=NMODE, 
                    nsec=NSEC, zmin=zmin, zmax=zmax, nz=nz, midline=midline, Psub=Psub, 
                    rho=rho, alpha=alpha, beta=beta, nInterp=nInterp, z0Interp=z0Interp, 
                    z1Interp=z1Interp, InputDir=InputDir,
                    data_index_start=data_index_start, data_index_end=data_index_end, 
                    data_step=data_step, num_shape_data=num_shape_data, flag_right=flag_right, i_ymin=i_ymin, i_ymax=i_ymax)

    surf, vol, corres_s, omega, frequency, eigen_vx, eigen_vy, eigen_vz, tm, \
        xm, zm, C1, C2 = \
            read_inputfiles(parameters)
    torch.set_default_tensor_type('torch.FloatTensor')

    C1 = C1.to(device)
    C2 = C2.to(device)

    k, e = find_included_indices()
	
    print('number of data = ', tm.size(0))
    print('batch size = ', parameters.batch_size)
    print('number of epochs = ', parameters.max_epochs)
    print('')
    sys.stdout.flush()
    surf.k = k
    surf.e_medial = e

    info = batchdata1(parameters)

    data = mydata()
    dataload = DataLoader(dataset=data, batch_size=1, shuffle=True)
    
    model = Neural_Network1().to(device)
	
    ### deleting data
    del vol
    del corres_s
    del omega
    del frequency
    del eigen_vx
    del eigen_vy
    del eigen_vz
    del tm
    del xm
    del zm
	
    start_time = time.time()

    train(device, model, 
        data_loader=dataload, info=info, surf=surf, 
        C1=C1, C2=C2, parameters=parameters)

    torch.save(model, cwd + "/model.pth")

    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    
	
