import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertConfig, BertTokenizer
import math

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

criterion = nn.BCELoss(size_average=True)
# criterion = nn.BCEWithLogitsLoss(size_average=True)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x

class CMSA(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(CMSA,self).__init__()
        n_heads = 1
        self.l1= nn.Linear(in_ch,out_ch)
        self.ese = eSEModule(out_ch)
        #self.conv1 = nn.Conv2d(int(out_ch),int(out_ch*n_heads),kernel_size= 1)
        #self.conv2 = nn.Conv2d(int(out_ch*n_heads),int(out_ch),kernel_size= 1)
        self.pool = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    def forward(self,x,embeddings,ch,vf_h = 10, vf_w = 10):

        embeddings = self.l1(embeddings)
        embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).repeat(1,1,vf_h,vf_h)
        print('embeddings: ',embedding.shape)
        #embeddings = embeddings.permute(0,2,1).reshape(-1,ch,vf_h,vf_w)
        n_heads = 1
        feats_fused = embeddings * x

        theta = self.ese(feats_fused)
        #theta = self.conv1(feats_fused)
        theta = theta.reshape(1,-1, int(ch*n_heads))

        phi = self.ese(feats_fused)
        #phi = self.conv1(feats_fused)
        phi = self.pool(phi)
        phi = phi.reshape(1,-1, int(ch*n_heads))
        phi = torch.transpose(phi, 2, 1)

        feat_nl = torch.bmm(theta, phi)
        feat_nl = nn.Softmax(dim = -1)(feat_nl)

        feats = self.ese(feats_fused)
        #feats = self.conv1(feats_fused)
        feats = self.pool(feats)
        feats = feats.reshape(1,-1,int(ch*n_heads))
        feats = torch.bmm(feat_nl,feats)
        feats = feats.reshape(-1,int(n_heads*ch),vf_h,vf_w)

        out = self.ese(feats)
        #out = self.conv2(feats)

        return out
    
class eSEModule2(nn.Module):
    def __init__(self, in_ch,out_ch, reduction=4):
        super(eSEModule2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return x
    
class CMSA2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(CMSA2,self).__init__()
        self.ese = eSEModule(in_ch)
        self.ese2 = eSEModule2(in_ch,out_ch)
        #self.conv1 = nn.Conv2d(int(out_ch),int(out_ch*n_heads),kernel_size= 1)
        #self.conv2 = nn.Conv2d(int(out_ch*n_heads),int(out_ch),kernel_size= 1)
        self.pool = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    def forward(self,x,ch,vf_h = 10, vf_w = 10):

        theta = self.ese(x)
        #theta = self.conv1(feats_fused)
        theta = theta.reshape(1,-1, int(ch))

        phi = self.ese(x)
        #phi = self.conv1(feats_fused)
        phi = self.pool(phi)
        phi = phi.reshape(1,-1, int(ch))
        phi = torch.transpose(phi, 2, 1)

        feat_nl = torch.bmm(theta, phi)
        feat_nl = nn.Softmax(dim = -1)(feat_nl)

        feats = self.ese(x)
        #feats = self.conv1(feats_fused)
        feats = self.pool(feats)
        feats = feats.reshape(1,-1,int(ch))
        feats = torch.bmm(feat_nl,feats)
        feats = feats.reshape(-1,int(ch),vf_h,vf_w)

        out = self.ese2(feats)
        #out = self.conv2(feats)

        return out


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

        self.conv1_1 = CMSA2(self.in_ch,64) ## n x c x 572 x 572
        self.conv1_2 = CMSA2(64,64)

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

        self.conv4_1d = CMSA2(1024,512)
        self.conv4_2d = CMSA2(512,512)
        #self.conv4_2d = REBNCONV(512,512)
        

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
        
        #combining visual and textual features
        self.bert = BertModel.from_pretrained("bert-base-uncased").cuda()
        #self.cmsa2 = CMSA(768,512)
        self.cmsa = CMSA(768,1024)

    def compute_loss(self, preds, targets):

        return muti_loss_fusion(preds,targets)

    def forward(self,x,captions):
        
        embeddings = []
        for caption in captions:
            inputs = tokenizer(caption, return_tensors="pt").to('cuda')
            outputs = self.bert(**inputs)
            embedding = outputs[0].squeeze(0)[0]
            embeddings.append(embedding)
        
        #embeddings = torch.tensor(self.bert.encode(captions)).cuda()
        embeddings = torch.tensor([item.cpu().detach().numpy() for item in embeddings]).cuda()
        embeddings = Variable(embeddings,requires_grad=True)

        x = self.conv1_1(x,ch = x.shape[1],vf_h = x.shape[2], vf_w = x.shape[3])
        x1 = self.conv1_2(x,ch = x.shape[1],vf_h = x.shape[2], vf_w = x.shape[3])

        x = self.pool1(x1)

        x = self.conv2_1(x)
        x2 = self.conv2_2(x)

        x = self.pool2(x2)

        x = self.conv3_1(x)
        x3 = self.conv3_2(x)

        x = self.pool3(x3)

        x = self.conv4_1(x)
        x4 = self.conv4_2(x)
        #x4 = x4 + self.cmsa2(x4,embeddings,ch = x4.shape[1],vf_h = x4.shape[2], vf_w = x4.shape[3])

        x = self.pool4(x4)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = x + self.cmsa(x,embeddings,ch = x.shape[1],vf_h = x.shape[2], vf_w = x.shape[3])
        
        x = self.up_conv_4(_upsample_like(x,x4))
        x = self.conv4_1d(torch.cat((x4,x),1),ch = torch.cat((x4,x),1).shape[1],vf_h = x.shape[2], vf_w = x.shape[3])
        x = self.conv4_2d(x, ch = x.shape[1],vf_h = x.shape[2], vf_w = x.shape[3])
        #x = x + self.cmsa2(x,embeddings,ch = x.shape[1],vf_h = x.shape[2], vf_w = x.shape[3])

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
