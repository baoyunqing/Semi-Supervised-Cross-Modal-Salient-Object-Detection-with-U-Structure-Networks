# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from skimage import exposure
#==========================dataset load==========================

class RescaleTV(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		h, w = image.shape[:2]

		# print("before flip",image.shape,label.shape)
		# print(image[0,0,0])
		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]
		# print(image[h-1,0,0])
		# print("after flip",image.shape,label.shape)

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label}

class RescaleTT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		# img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		# lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label}

class RandomPad(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self,sample):
		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		# h, w = image.shape[:2]

		row_pad = np.random.randint(0, self.output_size[0])
		col_pad = np.random.randint(0, self.output_size[1])

		# print("image.shape:",image.shape)
		# print("label.shape:",label.shape)

		img = np.pad(image,((row_pad,row_pad),(col_pad,col_pad),(0,0)),'constant')
		lbl = np.pad(label,((row_pad,row_pad),(col_pad,col_pad),(0,0)),'constant')


		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		# img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		# lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label}

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label}

class RescaleHW(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, caption, label, imname = sample['imidx'],sample['image'],sample['caption'],sample['label'],\
        sample['image_name']

		h, w = image.shape[:2]

		# if isinstance(self.output_size,int):
		# 	if h > w:
		# 		new_h, new_w = self.output_size*h/w,self.output_size
		# 	else:
		# 		new_h, new_w = self.output_size,self.output_size*w/h
		# else:
		# 	new_h, new_w = self.output_size
		#
		# new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size[0],self.output_size[1]),mode='constant')
		lbl = transform.resize(label,(self.output_size[0],self.output_size[1]),mode='constant', order=2, preserve_range=True)

		return {'imidx':imidx,'image':img, 'label':lbl, 'caption':caption, 'image_name':imname}

class RandomScale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		h, w = image.shape[:2]

		tar_idx = np.random.randint(0, len(self.output_size))
		tar_size = self.output_size[tar_idx]

		img = transform.resize(image,(tar_size,tar_size),mode='constant')
		# lbl = label
		lbl = transform.resize(label,(tar_size,tar_size),mode='constant', order=0, preserve_range=True)

		# if isinstance(self.output_size,int):
		# 	if h > w:
		# 		new_h, new_w = self.output_size*h/w,self.output_size
		# 	else:
		# 		new_h, new_w = self.output_size,self.output_size*w/h
		# else:
		# 	new_h, new_w = self.output_size
		#
		# new_h, new_w = int(new_h), int(new_w)
		#
		# # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# # img = transform.resize(image,(new_h,new_w),mode='constant')
		# # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
		#
		# img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		# lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx,'image':img, 'caption':caption, 'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx,'image':img, 'caption':caption, 'label':lbl}

# class RescaleRect(object):
#
# 	def __init__(self,oh,ow):
# 		assert isinstance(oh,(int,tuple))
# 		assert isinstance(ow,(int,tuple))
# 		self.oh = oh
# 		self.ow = ow
# 		# self.output_size = output_size
#
# 	def __call__(self,sample):
# 		imidx, image, label = sample['imidx'], sample['image'],sample['label']
#
# 		h, w = image.shape[:2]
#
# 		if isinstance(self.output_size,int):
# 			if h > w:
# 				new_h, new_w = self.output_size*h/w,self.output_size
# 			else:
# 				new_h, new_w = self.output_size,self.output_size*w/h
# 		else:
# 			new_h, new_w = self.output_size
#
# 		new_h, new_w = int(new_h), int(new_w)
#
# 		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
# 		img = transform.resize(image,(new_h,new_w),mode='constant')
# 		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
#
# 		return {'imidx':imidx, 'image':img,'label':lbl}

class CenterCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, caption, label=sample['imidx'],sample['image'],sample['caption'],sample['label']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		# print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
		assert((h >= new_h) and (w >= new_w))

		h_offset = int(math.floor((h - new_h)/2))
		w_offset = int(math.floor((w - new_w)/2))

		image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
		label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label}

# class RandomScale(object):
#
# 	def __init__(self,output_size):
# 		assert isinstance(output_size, (int, tuple))
# 		if isinstance(output_size, int):
# 			self.output_size = (output_size, output_size)
# 		else:
# 			assert len(output_size) == 2
# 			self.output_size = output_size
#
# 	def __call__(self,sample):
# 		imidx, image, label = sample['imidx'], sample['image'], sample['label']
#
# 		h, w = image.shape[:2]
# 		new_h, new_w = self.output_size
#
# 		top = np.random.randint(0, h - new_h)
# 		left = np.random.randint(0, w - new_w)
#
# 		image = image[top: top + new_h, left: left + new_w]
# 		label = label[top: top + new_h, left: left + new_w]
#
# 		return {'imidx':imidx,'image':image, 'label':label}

class RandomRotate(object):

	def __init__(self,rotate_flag):
		assert isinstance(rotate_flag, (int, tuple))
		if isinstance(rotate_flag, int):
			self.rotate_flag = (rotate_flag, rotate_flag)
		else:
			assert len(rotate_flag) == 2
			self.rotate_flag = rotate_flag

	def __call__(self,sample):
		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		if(self.rotate_flag[0]==self.rotate_flag[1]):
			r_degree = np.random.randint(1, 4)*90
		else:
			r_degree = np.random.randint(self.rotate_flag[0],self.rotate_flag[1])

		image = transform.rotate(image,r_degree,resize=True,preserve_range=True)
		label = transform.rotate(label,r_degree,resize=True,preserve_range=True)

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label}

class RandomFlip(object):

	def __init__(self,flip_flag):
		assert isinstance(flip_flag, (int, tuple))
		if isinstance(flip_flag, int):
			self.flip_flag = (flip_flag, flip_flag)
		else:
			assert len(flip_flag) == 2
			self.flip_flag = flip_flag

	def __call__(self,sample):
		imidx, image,label,caption, imname = sample['imidx'],sample['image'],sample['label'],sample['caption'],\
        sample['image_name']

		# random horizontal flip
		if(self.flip_flag[0]):
			if random.random() >= 0.5:
				image = image[::-1]
				label = label[::-1]

		# random vertical flip
		if(self.flip_flag[1]):
			if random.random() >= 0.5:
				image = image[:,::-1]
				label = label[:,::-1]

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label, 'image_name':imname}

# class RandomScale(object):
#
# 	def __init__(self,output_size):
# 		assert isinstance(output_size, (int, tuple))
# 		if isinstance(output_size, int):
# 			self.output_size = (output_size, output_size)
# 		else:
# 			assert len(output_size) == 2
# 			self.output_size = output_size
# 	def __call__(self,sample):
# 		imidx, image, label = sample['imidx'], sample['image'], sample['label']
#
# 		h, w = image.shape[:2]
# 		new_h, new_w = self.output_size
#
# 		top = np.random.randint(0, h - new_h)
# 		left = np.random.randint(0, w - new_w)
#
# 		image = image[top: top + new_h, left: left + new_w]
# 		label = label[top: top + new_h, left: left + new_w]
#
# 		return {'imidx':imidx,'image':image, 'label':label}

class RandomCrop(object):

    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self,sample):
        imidx, image, label, caption , imname= sample['imidx'],sample['image'],sample['label'],sample['caption'],\
        sample['image_name']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'caption':caption, 'image_name':imname}
    
class RandomContrast(object):

	def __init__(self,low_upper):
		assert isinstance(low_upper, (int, tuple))
		if isinstance(low_upper, int):
			self.output_contrast = (low_upper, low_upper)
		else:
			assert len(low_upper) == 2
			self.output_contrast = low_upper
	def __call__(self,sample):
		imidx, image, caption, label, imname = sample['imidx'],sample['image'],sample['caption'],sample['label'],\
        sample['image_name']

		# h, w = image.shape[:2]
		low, upper = self.output_contrast

		left = np.random.randint(0, low)
		right = np.random.randint(100-upper, 100)

		p2, p98 = np.percentile(image, (left, right))
		image = exposure.rescale_intensity(image, in_range=(p2, p98))

		return {'imidx':imidx,'image':image, 'caption':caption, 'label':label, 'image_name':imname}


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, caption, label = sample['imidx'],sample['image'],sample['caption'],sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'caption': caption, 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, caption, label, imname =sample['imidx'],sample['image'],sample['caption'],sample['label'],\
        sample['image_name']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx.copy()), 'image': torch.from_numpy(tmpImg.copy()), 'caption':caption, 'label': torch.from_numpy(tmpLbl.copy()), 'image_name':imname}

class SalObjDataset(Dataset):
    def __init__(self, img_name_list,lbl_name_list,caption_list,transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.caption_list = caption_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = io.imread(self.image_name_list[idx])#,img_num=0)
        imname = self.image_name_list[idx]
        caption = self.caption_list[idx]
        imidx = np.array([idx])

        if(0==len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])#,img_num=0)

        label = np.zeros(label_3.shape[0:2])
        if(3==len(label_3.shape)):
            label = label_3[:,:,0]
        elif(2==len(label_3.shape)):
            label = label_3

        if(3==len(image.shape) and 2==len(label.shape)):
            label = label[:,:,np.newaxis]
        elif(2==len(image.shape) and 2==len(label.shape)):
            image = image[:,:,np.newaxis]
            label = label[:,:,np.newaxis]

        sample = {'imidx':imidx, 'image':image, 'label':label, 'caption':caption, 'image_name':imname}

        if self.transform:
            sample = self.transform(sample)

        return sample

