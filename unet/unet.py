import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


criterion = nn.BCELoss(size_average=True)
# criterion = nn.BCEWithLogitsLoss(size_average=True)

def muti_loss_fusion(preds, target):
    loss = 0.0
    loss0 = criterion(preds[0],target)
    loss = loss + loss0

    for i in range(1,len(preds)):
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            tmp_target = _upsample_like(target,preds[i])
        loss = loss + criterion(preds[i],tmp_target)

    return loss0, loss

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,kernel_size=3,padding=1,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,kernel_size,padding=padding*dirate,dilation=dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        gps = int(out_ch/4)
        if(gps==0):
            gps = 1
        self.gn_s1 = nn.GroupNorm(gps, out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.gn_s1(self.conv_s1(hx)))

        return xout

class UNET(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(UNET,self).__init__()

        self.in_ch = 3
        self.out_ch = 1

        self.conv1_1 = REBNCONV(self.in_ch,64) ## n x c x 572 x 572
        self.conv1_2 = REBNCONV(64,64)

        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.conv2_1 = REBNCONV(64,128)
        self.conv2_2 = REBNCONV(128,128)

        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.conv3_1 = REBNCONV(128,256)
        self.conv3_2 = REBNCONV(256,256)

        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.conv4_1 = REBNCONV(256,512)
        self.conv4_2 = REBNCONV(512,512)

        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.conv5_1 = REBNCONV(512,1024)
        self.conv5_2 = REBNCONV(1024,1024)

        ## upsample
        self.up_conv_4 = REBNCONV(1024,512,kernel_size=1,padding=0)

        self.conv4_1d = REBNCONV(1024,512)
        self.conv4_2d = REBNCONV(512,512)

        self.up_conv_3 = REBNCONV(512,256)

        self.conv3_1d = REBNCONV(512,256)
        self.conv3_2d = REBNCONV(256,256)

        self.up_conv_2 = REBNCONV(256,128)

        self.conv2_1d = REBNCONV(256,128)
        self.conv2_2d = REBNCONV(128,128)

        self.up_conv_1 = REBNCONV(128,64)

        self.conv1_1d = REBNCONV(128,64)
        self.conv1_2d = REBNCONV(64,64)
        self.conv1_3d = nn.Conv2d(64,self.out_ch,1,padding=0)

    def compute_loss(self, preds, targets):

        return muti_loss_fusion(preds,targets)

    def forward(self,x,captions):

        x = self.conv1_1(x)
        x1 = self.conv1_2(x)

        x = self.pool1(x1)

        x = self.conv2_1(x)
        x2 = self.conv2_2(x)

        x = self.pool2(x2)

        x = self.conv3_1(x)
        x3 = self.conv3_2(x)

        x = self.pool3(x3)

        x = self.conv4_1(x)
        x4 = self.conv4_2(x)

        x = self.pool4(x4)

        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.up_conv_4(_upsample_like(x,x4))
        x = self.conv4_1d(torch.cat((x4,x),1))
        x = self.conv4_2d(x)

        x = self.up_conv_3(_upsample_like(x,x3))
        x = self.conv3_1d(torch.cat((x3,x),1))
        x = self.conv3_2d(x)

        x = self.up_conv_2(_upsample_like(x,x2))
        x = self.conv2_1d(torch.cat((x2,x),1))
        x = self.conv2_2d(x)

        x = self.up_conv_1(_upsample_like(x,x1))
        x = self.conv1_1d(torch.cat((x1,x),1))
        x = self.conv1_2d(x)
        x = self.conv1_3d(x)

        return [F.sigmoid(x)]
