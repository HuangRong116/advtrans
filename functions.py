import torch
import torch.nn as nn

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step,warmup_step=100):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_step = warmup_step

    # def step(self, current_step):
    #     if current_step <= self.decay_start_step:
    #         lr = self.start_lr
    #     elif current_step >= self.decay_end_step:
    #         lr = self.end_lr
    #     elif current_step <self.warmup_step:
    #         warmup_precent_done = current_step / self.warmup_step
    #         lr = self.start_lr * warmup_precent_done
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr
    #     else:
    #         #lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
    #         lr = self.start_lr ** 1.001
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr
    #     return lr
    
    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def Lossfunc(lossname,y,label,img,transimg,device,batch_size=1000):
    if lossname == "CrossEntropyLoss":
        loss = nn.CrossEntropyLoss()(y,label)
    elif lossname == "lsgan":
        real_label = torch.full((y.shape[0],y.shape[1]), 1., dtype=torch.float).to(device)
        loss = nn.MSELoss()(y, real_label)
    elif lossname == 'wgangp-mode':
        fake_image1, fake_image2 = transimg[:batch_size//2], transimg[batch_size//2:]
        z_1, z_2 = img[:batch_size//2], img[batch_size//2:]
        lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(torch.abs(z_2 - z_1))
        eps = 1 * 1e-5
        loss_lz = 1 / (lz + eps)
        #loss = -torch.mean(y) + loss_lz
        loss = nn.CrossEntropyLoss()(y,label)+loss_lz
    else:
        loss = -torch.mean(y)
    return loss