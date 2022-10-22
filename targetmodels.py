import torch

class CNN(torch.nn.Module):
    def __init__(self,cnn_drop_raito=0.5,in_c=3,out_c=10,kernel_size_conv=5,kernel_size_MaxPool2d=2,n_classes=10):
        super(CNN, self).__init__()
        
        # 3*32*32   --->   10*14*14
        self.cnn_drop_raito = cnn_drop_raito
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size_conv
        self.n_classes = n_classes
        self.kernel_size_MaxPool2d = kernel_size_MaxPool2d

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_c, self.out_c, self.kernel_size),
            torch.nn.BatchNorm2d(self.out_c),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(self.kernel_size_MaxPool2d)
            )
        
        #10*14*14   --->   20*5*5
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.out_c, 2*self.out_c,self.kernel_size),
            torch.nn.BatchNorm2d(2*self.out_c),
            torch.nn.ReLU(),
            )
    
        self.Dropout = torch.nn.Dropout(cnn_drop_raito)

    def forward(self,x):
        x = self.conv1(x)
        x = self.Dropout(x)
        x = self.conv2(x)
        x = x.view(1,-1)
        len = x.shape
        x = torch.nn.Linear(len[1],self.n_classes)(x)

        return x