import model_shared_weights as model
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import SupportingFunctions as sf
import sys

print('Training code has been started.')

### HYPERPARAMETERS
params = dict([('num_epoch', 200),
               ('batch_size', 1),
               ('learning_rate', 1e-3),
               ('num_workers', 0),          # It should be 0 for Windows machines
               ('exp_num', 7),              # CHANGE EVERYTIME
               ('save_flag', False),
               ('use_cpu', False),
               ('acc_rate', 4),
               ('K', 3)])   

### PATHS          
train_data_path  = 'Knee_Coronal_PD_RawData_300Slices_Train.h5'
train_coil_path  = 'Knee_Coronal_PD_CoilMaps_300Slices_Train.h5'
                 
# 0) Fix randomness for reproducible experiment
torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# 1) Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and (not(params['use_cpu']))) else 'cpu')

# 2) Load Data
dataset = sf.KneeDataset(train_data_path,train_coil_path, params['acc_rate'], num_slice=5)
loaders, datasets= sf.prepare_train_loaders(dataset,params,g)
mask = dataset.mask.to(device)

# 3) Create Model structure
denoiser = model.ResNet().to(device)
optimizer = torch.optim.Adam(denoiser.parameters(),lr=params['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

loss_arr       = np.zeros(params['num_epoch'])
loss_arr_valid = np.zeros(params['num_epoch'])

for epoch in range(params['num_epoch']):
    for i, (x0, kspace, mask_loss, mask_train, sens_map, index) in enumerate(loaders['train_loader']):
        x0      = x0.to(device)
        kspace = kspace.to(device)
        mask_loss = mask_loss.to(device)
        mask_train = mask_train.to(device)
        sens_map    = sens_map.to(device)
        # Forward pass
        xk = x0
        scale = torch.max(torch.abs(x0))
        for k in range(params['K']):
            L, zk = denoiser(xk/scale)
            xk = model.DC_layer(x0,zk*scale,L,sens_map,mask_train[0])
        ksp_loss = sf.encode(xk,sens_map,mask_loss[0])
        
        optimizer.zero_grad()
        # Loss calculation
        loss = sf.L1L2Loss(kspace*mask_loss[:,:,:,None], ksp_loss)
        
        if (torch.isnan(loss)):
            torch.save(denoiser.state_dict(), 'model_t_' + f'_SSDU_{epoch:03d}'+ '.pt')
            print ('-----------------------------')
            print (f'Epoch [{epoch+1}/{params["num_epoch"]}], \
                   loss: {loss:.08f}, \
                   index: {index}')
            print ('-----------------------------')
            torch.save(loss_arr, 'train_loss.pt')
            torch.save(loss_arr_valid, 'valid_loss.pt')
            sys.exit()
            
        loss_arr[epoch] += loss.item()/len(datasets['train_dataset'])
        loss.backward()
        
        # Optimize
        optimizer.step()
        
        if ((epoch+1)%10==0):
          torch.save(denoiser.state_dict(), 'model_t_' + f'_SSDU_{epoch+1:03d}'+ '.pt')
    
    for i, (x0, kspace, mask_loss, mask_train, sens_map, index) in enumerate(loaders['valid_loader']):
        with torch.no_grad():
            x0     = x0.to(device)
            kspace = kspace.to(device)
            mask_loss = mask_loss.to(device)
            mask_train = mask_train.to(device)
            sens_map   = sens_map.to(device)
            # Forward pass
            xk = x0
            for k in range(params['K']):
                L, zk = denoiser(xk)
                xk = model.DC_layer(x0,zk,L,sens_map,mask_train[0])
            ksp_loss = sf.encode(xk,sens_map,mask_loss[0])
            
            loss = sf.L1L2Loss(kspace*mask_loss[:,:,:,None], ksp_loss)
            loss_arr_valid[epoch] += loss.item()/len(datasets['valid_dataset'])
    
    scheduler.step()
    
    print ('-----------------------------')
    print (f'Epoch [{epoch+1}/{params["num_epoch"]}], \
           Loss training: {loss_arr[epoch]:.6f}, \
           Loss validation: {loss_arr_valid[epoch]:.6f}')
    print ('-----------------------------')

figure = plt.figure()
n = np.arange(1,params['num_epoch']+1)
plt.plot(n,loss_arr,n,loss_arr_valid)
plt.xlabel('epoch')
plt.title('Loss Graph')
plt.legend(['train loss', 'validation loss'])
figure.savefig('loss_graph.png')

torch.save(loss_arr, 'train_loss.pt')
torch.save(loss_arr_valid, 'valid_loss.pt')