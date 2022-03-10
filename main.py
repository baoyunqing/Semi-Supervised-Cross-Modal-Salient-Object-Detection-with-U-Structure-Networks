# train
# unet
#import os
from __future__ import print_function
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
# from tensorboard_logger import configure, log_value

# from data_loader import Rescale
# from data_loader import RescaleT
from data_loader import RescaleHW
from data_loader import RandomFlip
from data_loader import RandomCrop
from data_loader import RandomContrast
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from data_loader import RandomRotate

import time

from unet import U2NET, UNET, BASNet
from basics import normPRED, f1score, PRF1Scores, compute_IoU, compute_mae

'''
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
'''
#writer = SummaryWriter('./log')


def _upsample_like(src,tar):
    if(src.shape[2:] != tar.shape[2:]):
        src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src

# import pytorch_ssim
# import pytorch_iou
# import pytorch_dice

def get_im_gt_name_list(im_dirs, gt_dirs, flag='train'):
    img_name_list = []
    lbl_name_list = []
    caption_list = []
    
    for i in range(len(im_dirs)):
        if flag == 'train':
            df = pd.read_csv(im_dirs[i].replace('im','')+'DUTS-TR.csv')
        else:
            df = pd.read_csv(im_dirs[i].replace('im','')+'DUTS-TE.csv')
            
        tmp_im_list = glob(im_dirs[i]+'/*.jpg')
        for tmp_im in tmp_im_list:
            img_name_list.append(tmp_im)
            
            index = df.index[df['image_name'] == tmp_im.split('/')[-1]][0]
            caption = df.iloc[index]['caption']
            caption_list.append(caption)

        if(gt_dirs==[]):
            continue

        tmp_gt_list = [gt_dirs[i]+'/'+x.split('/')[-1][0:-4]+'.png' for x in tmp_im_list]
        if(flag=='train'):
            lbl_name_list.extend(tmp_gt_list)
        else:
            lbl_name_list.extend(tmp_gt_list)

        print("---",i,"/",len(im_dirs),"---")
        print('-im-',im_dirs[i],': ',len(tmp_im_list))
        print('-gt-',gt_dirs[i],': ',len(tmp_gt_list))
        
        #print(img_name_list[101],lbl_name_list[101],caption_list[101])

    return img_name_list, lbl_name_list, caption_list

def get_dataloaders_val(img_name_list, lbl_name_list, caption_list, imsize=(320,320), batch_size_val=1):
    dataloaders = []
    salobj_dataset_val = SalObjDataset(
            img_name_list=img_name_list,
            lbl_name_list=lbl_name_list,
            caption_list = caption_list,
            transform=transforms.Compose([
                RescaleHW(imsize),
                ToTensorLab(flag=0)]))
    salobj_dataloader_val = DataLoader(salobj_dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=1)
    dataloaders.append(salobj_dataloader_val)
    print('length of dataloaders: ', len(dataloaders))
    return dataloaders

## def the loss functions
# criterion = nn.MSELoss(size_average=True)
#criterion = nn.BCEWithLogitsLoss(size_average=True)
criterion = nn.BCELoss(size_average=True)
#Module: pytorch_ssim.SSIM(window_size=11,size_average=True)
# ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True) ### Log SSIM
# iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

    # iou_out = iou_loss(pred,target)
    # ssim_out = 1 - ssim_loss(pred,target)
    bce_out = criterion(pred,target)

    loss = bce_out# + iou_out + ssim_out
    # loss = bce_out
    #print("\n BCE: %3f, IOU: %3f, SSIM: %3f, SUM: %3f"%(bce_out.data[0],iou_out.data[0],ssim_out.data[0], loss.data[0]))
    return loss

def muti_bce_loss_fusion(ds, labels_v):
    loss = 0.0
    # ws = [0.16, 0.4, 1.0]
    # print("len(ds) ", len(ds))
    for i in range(len(ds)):
        # tmp = _upsample_like(ds[i],labels_v)
        # print(i,"ds[i] shape: ", ds[i].shape)
        # print("labels shape: ", labels_v.shape)
        loss = loss + bce_ssim_loss(ds[i], labels_v)
        # losses.append(bce_ssim_loss(ds[i], labels_v))
    # loss = np.sum(losses)
    loss0 = bce_ssim_loss(ds[0], labels_v)#losses[0]

    # losses_str = [str(x.data[0]) for x in losses]
    # print("--"+'_'.join(losses_str))
    # print("---loss: ",loss.data[0], "tar_loss:", loss0.data[0])
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))

    return loss0, loss

def main():
    ## configure model save directory
    model_dir = "saved_models/basnet"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    ## hyperparamters_itr_32000_traLoss_0.4076_traTarLoss_0.0484_valLoss_1.0967_valTarLoss_0.15_maxF1_0.8643_mae_0.0558_time_0.019108

    epoch_num = 1000
    im_height = 320
    im_width = 320
    crop_size = 288
    batch_size_train = 1
    batch_size_val = 1
    train_num = 0
    val_num = 0

    ## configure the  train and validation datasets
    tra_im_dirs =  ['dataset/DUTS/DUTS-TR/im']
                   # '../dataset_from_sinet/TrainDataset/CPD1K/Imgs',
                   # '../dataset_from_sinet/TrainDataset/CAMO/Imgs']
    tra_gt_dirs =  ['dataset/DUTS/DUTS-TR/gt']
                    #['dataset/DUTS/DUTS-TR/gt']
                    # '../dataset_from_sinet/TrainDataset/CPD1K/GT',
                    #'../dataset_from_sinet/TrainDataset/CAMO/GT']

    val_im_dirs = ['dataset/DUTS/DUTS-TE/im']
                  #['dataset/DUTS/DUTS-TE/im',]
                  #'../dataset/SOD/img']
                  # '/data3/xuebin/HRSOD/CascadePSP/data/BIG_im_gt/test_val_cmb/im']
                  # '/data3/xuebin/HRSOD/CascadePSP/data/hr-sod/val/im']#'../dataset_from_sinet/TestDataset/COD10K/Imgs',
                    #'../dataset_from_sinet/TestDataset/CPD1K/Imgs',
                    #'../dataset_from_sinet/TestDataset/CAMO/Imgs',
                    #'../dataset_from_sinet/TestDataset/CHAMELEON/Imgs']
    val_gt_dirs = ['dataset/DUTS/DUTS-TE/gt']
                  #['dataset/DUTS/DUTS-TE/gt',]
                  #'../dataset/SOD/gt']
                  # '/data3/xuebin/HRSOD/CascadePSP/data/BIG_?im_gt/test_val_cmb/gt']#,
                  # '/data3/xuebin/HRSOD/CascadePSP/data/hr-sod/val/gt']#'../dataset_from_sinet/TestDataset/COD10K/GT',
                    #'../dataset_from_sinet/TestDataset/CPD1K/GT',
                    #'../dataset_from_sinet/TestDataset/CAMO/GT',
                    #'../dataset_from_sinet/TestDataset/CHAMELEON/GT']

    ## extracting the training image and ground truth pathes from the given directories
    print("=====Extracting Training image and ground truth=====")
    tra_img_name_list, tra_lbl_name_list, tra_caption_list = get_im_gt_name_list(tra_im_dirs,tra_gt_dirs,flag='train')
    print("----- total training --im:",len(tra_img_name_list),"--gt: ", len(tra_lbl_name_list))
    print("=====Extracting Validation image and ground truth=====")
    val_img_name_list, val_lbl_name_list, val_caption_list = get_im_gt_name_list(val_im_dirs,val_gt_dirs,flag='val')
    print("----- total validation --im:",len(val_img_name_list),"--gt: ", len(val_lbl_name_list))
    print("------------------------------------------------------")

    train_num = len(tra_img_name_list)
    # val_num = len(val_img_name_list)

    ## define the training and validation dataloder
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        caption_list = tra_caption_list,
        # transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(224)]))
        transform=transforms.Compose([
            RescaleHW((im_height,im_width)), #320
            RandomFlip((1,1)), # both horizontal and vertical flip
            RandomCrop(crop_size), 
            RandomContrast((10,10)),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    ## get dataloaders for multiple validation datasets
    val_dataloders = get_dataloaders_val(val_img_name_list, val_lbl_name_list, val_caption_list, imsize=(im_height,im_width))


    ## define the model
    print("---define model...")
    #net = build_model(base_model_cfg='vgg')
    net = BASNet(3,1)
    net.load_state_dict(torch.load(model_dir + '/itr_28500_traLoss_0.1971_traTarLoss_0.0163_valLoss_1.6457_valTarLoss_0.182_maxF1_0.8844_mae_0.0409_time_0.011197.pth'),strict = True)
    '''
    for k,v in net.named_parameters():
        if k == 'self.bert':
            v.requires_grad = False
        else:
            v.requires_grad = True
    '''
                       
    if torch.cuda.is_available():
        net.cuda()
        #net = torch.nn.DataParallel(net,device_ids= device_ids)

    ## define the optimizer
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.2)
    scheduler_lr =optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100,T_mult=2)
    #optimizer = torch.nn.DataParallel(optimizer,device_ids = device_ids)

    ## training hyperparamter1
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frequ = 2000
    last_f1 = [0 for x in range(len(val_im_dirs))]


    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, captions, labels = data['image'], data['caption'], data['label']
            #print('length of input', len(data['image']))
            
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
                       
            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v,labels_v = Variable(inputs.cuda(),requires_grad=False), Variable(labels.cuda(),requires_grad=False)
            else:
                inputs_v,labels_v = Variable(inputs,requires_grad=False),Variable(labels,requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            ds = net(inputs_v,captions)
            
            '''
            img = ds[0].cpu().detach().numpy()
            img = img[0].squeeze(0)
            print(captions)
            cv2.imshow('test',img)
            cv2.waitKey(0)
            cv2.imwrite('example.jpg',img)
            '''
            
            loss2, loss = muti_bce_loss_fusion(ds, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()
            

            # del outputs, loss
            del ds, loss2, loss

            print(">>>"+model_dir.split('/')[-1]+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            
            

            if ite_num % save_frequ == 0:  # validate every 2000 iterations
                print("Validating...")
                net.eval()

                val_loss = 0.0
                tar_loss = 0.0
                val_cnt = 0.0

                tmp_f1 = []
                tmp_mae = []
                tmp_time = []
                for k in range(len(val_dataloders)):

                    salobj_dataloader_val = val_dataloders[k]

                    val_num = len(val_lbl_name_list)
                    mybins = np.arange(0,256)
                    PRE = np.zeros((val_num,len(mybins)-1))
                    REC = np.zeros((val_num,len(mybins)-1))
                    F1 = np.zeros((val_num,len(mybins)-1))
                    MAE = np.zeros((val_num))


                    for i_val, data_val in enumerate(salobj_dataloader_val):
                        val_cnt = val_cnt + 1.0
                        # if(i_val>10):
                        #     break

                        inputs_val, labels_val, captions_val, imidx_val = data_val['image'], data_val['label'], data_val['caption'], data_val['imidx']
                        

                        inputs_val = inputs_val.type(torch.FloatTensor)
                        labels_val = labels_val.type(torch.FloatTensor)  # torch.LongTensor
                        #embeddings_val = torch.tensor(embeddings_val)

                        # wrap them in Variable
                        if torch.cuda.is_available():
                            inputs_val_v, labels_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(labels_val.cuda(), requires_grad=False)
                        else:
                            inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,requires_grad=False)
                            
                        # outputs_val = net(inputs_val_v)
                        # loss_val = criterion(outputs_val, labels_val_v)
                        t_start = time.time()
                        ds_val = net(inputs_val_v,captions_val)
                        t_end = time.time()-t_start
                        tmp_time.append(t_end)

                        loss2_val, loss_val = muti_bce_loss_fusion(ds_val, labels_val_v)

                        # compute F measure
                        pred_val = ds_val[0][:,0,:,:] # d1 other than d0
                        pred_val = normPRED(pred_val)
                        pre,rec,f1,mae  = PRF1Scores(pred_val, val_lbl_name_list, imidx_val, '', mybins)
                        # m = compute_mae(pred_val,)

                        i_test = imidx_val.data.numpy()[0][0]
                        PRE[i_test,:]=pre
                        REC[i_test,:] = rec
                        F1[i_test,:] = f1
                        MAE[i_test] = mae

                        # if(loss_val.data[0]>1):
                        val_loss += loss_val.item()#data[0]
                        tar_loss += loss2_val.item()#data[0]

                        print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f"% (i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1[i_test,:]), MAE[i_test],t_end))

                        # del outputs_val, loss_val
                        del ds_val, loss2_val, loss_val

                    print('============================')
                    PRE_m = np.mean(PRE,0)
                    REC_m = np.mean(REC,0)
                    f1_m = (1+0.3)*PRE_m*REC_m/(0.3*PRE_m+REC_m)
                    tmp_f1.append(np.amax(f1_m))
                    tmp_mae.append(np.mean(MAE))
                    # print("The max F1 Score: %f"%(np.max(f1_m)))

                print('[epoch: %3d/%3d, ite: %5d] tra_ls: %3f, val_ls: %3f, tar_ls: %3f, maxf1: %3f'% (epoch + 1, epoch_num, ite_num, running_loss / ite_num4val, val_loss/val_cnt, tar_loss/val_cnt, tmp_f1[-1]))
                

                tmp_out = 0
                print("last_f1:",last_f1)
                print("tmp_f1:",tmp_f1)
                for fi in range(len(last_f1)):
                    if(tmp_f1[fi]>last_f1[fi]):
                        tmp_out = 1
                print("tmp_out:",tmp_out)
                if(tmp_out):
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x,4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx,4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/itr_"+str(ite_num)+\
                                "_traLoss_"+str(np.round(running_loss / ite_num4val,4))+\
                                "_traTarLoss_"+str(np.round(running_tar_loss / ite_num4val,4))+\
                                "_valLoss_"+str(np.round(val_loss /(i_val+1),4))+\
                                "_valTarLoss_"+str(np.round(tar_loss /(i_val+1),4)) + \
                                "_maxF1_" + maxf1 + \
                                "_mae_" + meanM + \
                                "_time_" + str(np.round(np.mean(np.array(tmp_time)),6))+".pth"
                    torch.save(net.state_dict(), model_dir + model_name)
            
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
                
        scheduler_lr.step()
        print('learning rate: ', scheduler_lr.get_lr())
                # exit()
    #model.add_graph(net,(inputs,))
    print('-------------Congratulations! Training Done!!!-------------')

if __name__ == "__main__":
    main()
