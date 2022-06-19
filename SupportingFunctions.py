import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader

### DEFINE FFT2 AND IFFT2 FUNCTIONS
# y = FFT(x): FFT of one slice image to kspace: [1 Nx Ny Nc] --> [1 Nx Ny Nc]
def fft2 (image, axis=[1,2]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image, dim=axis), dim=axis, norm='ortho'), dim=axis)

# x = iFFT(y): iFFT of one slice kspace to image: [1 Nx Ny Nc] --> [1 Nx Ny Nc]
def ifft2 (kspace, axis=[1,2]):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=axis), dim=axis, norm='ortho'), dim=axis)

# y = Ex: encoding one slice image to kspace: [1 Nx Ny] --> [1 Nx Ny Nc]
# S: sensitivity map
def encode(x,S,mask):
    if mask==None:
        return fft2(S*x[:,:,:,None])
    else:
        return fft2(S*x[:,:,:,None])*mask[None,:,:,None]

# y = E'x: reconstruction from kspace to image space: [1 Nx Ny Nc] --> [1 Nx Ny]
# S: sensitivity map
def decode(x,S):
    return torch.sum(ifft2(x)*torch.conj(S), axis=3)

# Normalised Mean Square Error (NMSE)
# gives the nmse between x and xref
def nmse(x,xref):
    return np.sum((x-xref)**2) / np.sum((xref)**2)

class KneeDatasetTrain():
    def __init__(self,data_path,coil_path,R,num_slice,loss_train_loss=0.4,num_ACS=24):
        f = h5py.File(data_path, "r")
        start_slice = 0
        r = 1
        self.kspace    = f['kspace'][start_slice:start_slice+num_slice*r:r]
        self.kspace    = torch.from_numpy(self.kspace)
        
        self.n_slices  = self.kspace.shape[0]
        
        S = h5py.File(coil_path, "r")
        _, value = list(S.items())[0]
        self.sens_map    = value[start_slice:start_slice+num_slice*r:r]
        self.sens_map    = torch.from_numpy(self.sens_map)
        
        self.mask = torch.zeros((self.kspace.shape[1],self.kspace.shape[2]), dtype=torch.cfloat)
        self.mask[:,::R] = 1.0
        self.mask[:,(self.kspace.shape[2]-num_ACS)//2:(self.kspace.shape[2]+num_ACS)//2] = 1.0
        
        self.gauss_kernel = gauss_gen(self.mask.shape[0], self.mask.shape[1], sigma=1.0)
        
        #self.kspace = self.kspace*self.mask[None,:,:,None]
        self.x0   = torch.empty(self.kspace.shape[0:3], dtype=torch.cfloat)
        self.R    = 1/(torch.abs(self.mask).sum()/(self.kspace.shape[1]*self.kspace.shape[2]))
        self.mask_loss = torch.zeros(self.kspace.shape[0:3], dtype=torch.cfloat)
        self.mask_train = torch.zeros(self.kspace.shape[0:3], dtype=torch.cfloat)
        
        for i in range(self.kspace.shape[0]):
            self.random = torch.rand((self.kspace.shape[1],self.kspace.shape[2]))
            # 0.43 --> mask_loss / mask = 0.4
            self.gauss_mask = (self.random * self.gauss_kernel) > 0.43
            self.mask_loss[i] = self.gauss_mask * self.mask
            self.mask_loss[i,158:162,182:186] = 0.0 #4x4 small ACS area
            
            self.mask_train[i] = self.mask - self.mask_loss[i]
            
            self.kspace[i] = self.kspace[i:i+1]*self.mask[None,:,:,None]
            scale = torch.max(torch.abs(self.kspace[i]))
            self.kspace[i] = self.kspace[i] / scale
            
            self.x0[i] = decode(self.kspace[i:i+1]*self.mask_train[i,:,:,None],self.sens_map[i:i+1])
            
    def __getitem__(self,index):
        return self.x0[index], self.kspace[index], self.mask_loss[index], self.mask_train[index], self.sens_map[index], index
    def __len__(self):
        return self.n_slices  

class KneeDatasetTest():
    def __init__(self,data_path,coil_path,R,num_slice,loss_train_loss=0.4,num_ACS=24):
        f = h5py.File(data_path, "r")
        start_slice = 10
        r = 40
        self.kspace    = f['kspace'][start_slice:start_slice+num_slice*r:r]
        self.kspace    = torch.from_numpy(self.kspace)
        
        self.n_slices  = self.kspace.shape[0]
        
        S = h5py.File(coil_path, "r")
        _, value = list(S.items())[0]
        self.sens_map    = value[start_slice:start_slice+num_slice*r:r]
        self.sens_map    = torch.from_numpy(self.sens_map)
        
        self.mask = torch.zeros((self.kspace.shape[1],self.kspace.shape[2]), dtype=torch.cfloat)
        self.mask[:,::R] = 1.0
        self.mask[:,(self.kspace.shape[2]-num_ACS)//2:(self.kspace.shape[2]+num_ACS)//2] = 1.0
        
        #self.kspace = self.kspace*self.mask[None,:,:,None]
        self.x0   = torch.empty(self.kspace.shape[0:3], dtype=torch.cfloat)
        self.xref = torch.empty(self.kspace.shape[0:3], dtype=torch.cfloat)
        self.R    = 1/(torch.abs(self.mask).sum()/(self.kspace.shape[1]*self.kspace.shape[2]))
        
        for i in range(self.kspace.shape[0]):
            self.kspace[i] = (self.kspace[i:i+1] / 
                              torch.max(torch.abs(self.kspace[i:i+1]*self.mask[None,:,:,None])))
            
            self.x0[i] = decode(self.kspace[i:i+1]*self.mask[None,:,:,None],self.sens_map[i:i+1])
            
            self.xref[i] = decode(self.kspace[i:i+1],self.sens_map[i:i+1])
            
    def __getitem__(self,index):
        return self.x0[index], self.kspace[index], self.sens_map[index], self.xref[index], index
    def __len__(self):
        return self.n_slices


# Gaussian kernel generator
def gauss_gen(Nx, Ny, sigma):
    xs = torch.linspace(-1, 1, steps=Nx)
    ys = torch.linspace(-1, 1, steps=Ny)
    x, y = torch.meshgrid(xs, ys)
    z = (x*x + y*y)/(2*sigma)
    z = torch.exp(-z)
    z = z / torch.max(z)
    return z


# complex 1 channel to real 2 channels
def ch1to2(data1):       
    return torch.cat((data1.real,data1.imag),0)
# real 2 channels to complex 1 channel
def ch2to1(data2):       
    return data2[0:1,:,:] + 1j * data2[1:2,:,:] 

def prepare_train_loaders(dataset,params,g):
    train_num  = int(dataset.n_slices * 0.8)
    valid_num  = dataset.n_slices - train_num

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_num,valid_num],  generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset       = train_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)

    valid_loader = DataLoader(dataset       = valid_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)

    full_loader= DataLoader(dataset         = dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = False,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)
    
    datasets = dict([('train_dataset', train_dataset),
                     ('valid_dataset', valid_dataset)])  
    
    loaders = dict([('train_loader', train_loader),
                    ('valid_loader', valid_loader),
                    ('full_loader', full_loader)])

    return loaders, datasets

def prepare_test_loaders(test_dataset,params):
    test_loader  = DataLoader(dataset       = test_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'])
    
    datasets = dict([('test_dataset', test_dataset)])  
    
    loaders = dict([('test_loader', test_loader)])

    return loaders, datasets

# Normalised L1-L2 loss calculation
# loss = normalised L1 loss + normalised L2 loss
def L1L2Loss(ref, recon):
    return torch.norm(recon-ref,p=1)/torch.norm(ref,p=1) + torch.norm(recon-ref,p=2)/torch.norm(ref,p=2)

