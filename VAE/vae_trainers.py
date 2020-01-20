import sys
import numpy as np
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.ndimage.interpolation import rotate

def rotate_tensor(input, angles):
    input = input.cpu()
    """ Somewhat hacky. Needs to be paralelized and implemented in pytorch!"""
    output = []
    for i in range(input.shape[0]): #basically batch_sizs
        out = rotate(input[i, ...], 360*angles[i], axes=(1, 2), reshape=False)
        output.append(out)
    return torch.FloatTensor(np.stack(output, 0))

def compact_extended_loss(target, output, device):
    #########    Compact vs Extended    #########
    extended = target > 1
    extended = Variable(extended.float().to(device), requires_grad=False)
    pred_ext = torch.sigmoid(output)
    ext_loss = F.binary_cross_entropy(pred_ext, extended, reduction='sum').div(len(pred_ext))

    return ext_loss

def fri_frii_loss(target, output, device):
    o = torch.sigmoid(output)[target > 1]
    c = target[target > 1]

    o = o[c < 4] # FRI == 2, FRII == 3
    c = c[c < 4]

    c = Variable( (c == 3).float().to(device), requires_grad=False)
    fr_loss = F.binary_cross_entropy(o, c, reduction='sum').div(len(c))

    return fr_loss

#################################
#####      Train Steps      #####
#################################
def train_step_b_vae_cap(model, device, data_loader, optim, epoch, loss_fun, log_interval=5, learn_rot=False):
    model.train()
    s = ''
    r_loss = 0
    batch_sum = 0
    avg_r_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_sum += len(data)
        data = data.to(device)
        target = Variable(target, requires_grad=False).to(device)
        #Forward Pass
        optim.zero_grad()
        output = model(data)
        
        #############################################
        ######### check against known class #########
        #############################################
        #########    Compact vs Extended    #########
        ext_loss = compact_extended_loss(target, output[1][:,0], device)
        # # #########       FRI vs FRII         #########
        fr_loss = fri_frii_loss(target, output[1][:, 1], device)
        # #########   Try to learn rotation   #########
        # if learn_rot :
        #   rot = torch.sigmoid(output[1][:,-1])
        #   data = rotate_tensor(data, rot).to(device)

        # BCE Loss
        c, r_loss , g_loss = loss_fun(output, data)
        loss = r_loss + g_loss + 20 * (ext_loss + fr_loss)#
        avg_r_loss += r_loss.item()
        #Backpropagation
        loss.backward()
        optim.step()
        s = 'Train Epoch: {:3d} [{:5d}/{:5d} ({:3.0f}%)]   Loss: {:4.4f}   R_Loss: {:4.4f}   Capacity: {:4.2f}'
        s = s.format(epoch, batch_sum, len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()/len(data), r_loss.item()/len(data), c)
        if batch_idx % log_interval == 0:
            sys.stdout.write('{}\r'.format(s))
            sys.stdout.flush()
    return s, avg_r_loss / batch_sum

def train_step_b_vae(model, device, data_loader, optim, epoch, loss_fun, log_interval=5, learn_rot=False):
    model.train()
    s = ''
    r_loss = 0
    batch_sum = 0 
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_sum += len(data)
        data = data.to(device)
        target = Variable(target, requires_grad=False).to(device)
        #Forward Pass
        optim.zero_grad()
        output = model(data)
        
        #############################################
        ######### check against known class #########
        #############################################
        #########    Compact vs Extended    #########
        ext_loss = compact_extended_loss(target, output[1][:,0], device)
        # #########       FRI vs FRII         #########
        fr_loss = fri_frii_loss(target, output[1][:, 1], device)
        #########   Try to learn rotation   #########
        if learn_rot :
          rot = torch.sigmoid(output[1][:,-1])
          data = rotate_tensor(data, rot).to(device)

        # BCE Loss
        r_loss , g_loss = loss_fun(output, data)
        loss = r_loss + g_loss + 40 * (ext_loss + fr_loss)# + reg_loss)
        #Backpropagation
        loss.backward()
        optim.step()
        s = 'Train Epoch: {:3d} [{:5d}/{:5d} ({:3.0f}%)]   Loss: {:4.4f}   R_Loss: {:4.4f}'
        s = s.format(epoch, batch_sum, len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item(), r_loss.item())
        if batch_idx % log_interval == 0:
            sys.stdout.write('{}\r'.format(s))
            sys.stdout.flush()
    return s, r_loss.item()

#################################
#####       Test Steps      #####
#################################
def test_step_b_vae_cap(model, device, data_loader, loss_fun, learn_rot=False):
    model.train()
    r_loss = 0
    avg_r_loss = 0
    batch_sum = 0 
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_sum += len(data)
        with torch.no_grad():
            data = data.to(device)
            #Forward Pass
            output = model(data)
            #########   Use learned rotation   #########
            if learn_rot :
              rot = torch.sigmoid(output[1][:,-1])
              data = rotate_tensor(data, rot).to(device)
            # BCE Loss
            c, r_loss , g_loss = loss_fun(output, data)
            avg_r_loss += r_loss.item()
    return avg_r_loss / batch_sum

def test_step_b_vae(model, device, data_loader, loss_fun, learn_rot=False):
    model.train()
    r_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            data = data.to(device)
            #Forward Pass
            output = model(data)
            #########   Use learned rotation   #########
            if learn_rot :
              rot = torch.sigmoid(output[1][:,-1])
              data = rotate_tensor(data, rot).to(device)
            # BCE Loss
            r_loss , g_loss = loss_fun(output, data)
    return r_loss.item()