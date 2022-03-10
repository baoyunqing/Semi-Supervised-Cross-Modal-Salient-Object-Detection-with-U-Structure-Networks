import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

# from data_loader import RescaleT
# from data_loader import CenterCrop
# # from data_loader import RandomCrop
# from data_loader import ToTensor
# from data_loader import ToTensorLab
# from data_loader import SalObjDataset
#
# from unet import myResUNetSOTE97Q
import time

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def f1score(pd,gt,mybins):

	# for i in range(0,255):
	gtNum = gt[gt>128].size
	# plt.imshow(pd)
	# plt.show()
	# print(np.amax(pd))
	pp = pd[gt>128]
	nn = pd[gt<=128]
	# print(max(pp))
	# print(max(nn))
	# print("=====")
	pp_hist,pp_edges = np.histogram(pp,bins=mybins)
	nn_hist,nn_edges = np.histogram(nn,bins=mybins)

	# print(pp_hist)
	# print(nn_hist)
	# print("------")

	pp_hist_flip = np.flipud(pp_hist)
	nn_hist_flip = np.flipud(nn_hist)

	pp_hist_flip_cum = np.cumsum(pp_hist_flip)
	nn_hist_flip_cum = np.cumsum(nn_hist_flip)

	precision = (pp_hist_flip_cum + 1e-4)/(pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
	# print(pp_hist_flip_cum)
	# print(nn_hist_flip_cum)
	recall = (pp_hist_flip_cum + 1e-4)/(gtNum + 1e-4)

	f1 = (1+0.3)*precision*recall/(0.3*precision+recall)

	return np.reshape(precision,(1,len(precision))),np.reshape(recall,(1,len(recall))),np.reshape(f1,(1,len(f1)))


# def PRF1Scores(d91, labels_val, val_lbl_name_list, imidx_val, '', mybins):
def PRF1Scores(d,lbl_name_list,imidx_val,d_dir,mybins):

    predict = d
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()


    im = Image.fromarray(predict_np*255).convert('RGB')

    # # image = io.imread(img_name_list[i_test])
    # # imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    # # pb_np = np.array(imo)#np.resize(predict_np*255,(image.shape[0],image.shape[1]))
	#
    # img_name = lbl_name_list[i_test].split("/")[-1]
    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1,len(bbb)):
    #     imidx = imidx + "." + bbb[i]
	#
    # #ground truth image idx
    # gt_name = data_dir + groundtruth_dir + imidx + ".png"
    # gt = io.imread(gt_name)

    i_test = imidx_val.data.numpy()[0]
    gt = io.imread(lbl_name_list[i_test[0]])
    if len(gt.shape)>2:
        gt = gt[:,:,0]

    imo = im.resize((gt.shape[1],gt.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)#np.resize(predict_np*255,(image.shape[0],image.shape[1]))

    # #img_name_wo_ext = img_name.split(".")[0].split("_")[0]
    # #print("output as: %s"%(data_dir+d_dir+imidx+'.png'))
    # imo.save(out_dir+prediction_dir+d_dir+imidx+'.png')###################
    # #im.save(data_dir+d_dir+img_name.split(".")[0]+'.png')
    pb_np255 = (pb_np[:,:,0]-np.amin(pb_np[:,:,0]))/(np.amax(pb_np[:,:,0])-np.amin(pb_np[:,:,0]))*255
    # print(np.amax(pb_np255))
    pre, rec, f1 = f1score(pb_np255,gt,mybins)
    mae = compute_mae(pb_np255,gt)

    return pre, rec, f1, mae

def compute_IoU(d,lbl_name_list,imidx_val,d_dir,mybins):
    predict = d
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    i_test = imidx_val.data.numpy()[0]
    gt = io.imread(lbl_name_list[i_test[0]])
    if len(gt.shape)>2:
        gt = gt[:,:,0]

    imo = im.resize((gt.shape[1],gt.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)#np.resize(predict_np*255,(image.shape[0],image.shape[1]))

    pb_np = (pb_np[:,:,0]+1e-8)/(np.amax(pb_np[:,:,0])+1e-8)
    gt = (gt+1e-8)/(np.amax(gt)+1e-8)

    pb_bw = pb_np > 0.5
    gt_bw = gt > 0.5

    pb_and_gt = np.logical_and(pb_bw,gt_bw)
    numerator = np.sum(pb_and_gt.astype(np.float))+1e-8
    demoninator = np.sum(pb_bw.astype(np.float))+np.sum(gt_bw.astype(np.float))-numerator+1e-8

    return numerator/demoninator

def compute_mae(mask1,mask2):
    h,w=mask1.shape
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError/(float(h)*float(w)*255.0)

    return maeError
