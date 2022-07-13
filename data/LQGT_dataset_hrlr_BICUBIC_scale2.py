import os
import numpy as np
import cv2
import random
from math import ceil
from torch.utils.data import Dataset
import torch.nn.functional as F


class Testdataset(Dataset):
    def __init__(self, path_gt, path_lq, transform, patch_size = 64, scale=2, crop_mode = False):
        super(Testdataset, self).__init__()
        self.transform = transform
        # self.dirname_GT = '/home/mhkim/real-world-sr/datasets/DIV2K/DIV2K_valid_LR/X4'
        # self.dirname_GT = '/home/mhkim/real-world-sr/datasets/DIV2K/DIV2K_valid_LR_unknown/X4'
        # self.dirname_GT = '/home/mhkim/real-world-sr/datasets/DIV2K/DIV2K_valid_HR'
        # self.dirname_LQ = '/home/mhkim/real-world-sr/datasets/DIV2K/generated/from_CycleGAN_lrlr_unknown/valid_cyclegan_unknown_trainis256crop_400lr400lr'
#         self.dirname_GT = '/media/data1/mhkim/DS/DIV2K/DIV2K_valid_HR_'
#         self.dirname_LQ = '/media/data1/mhkim/DS/DIV2K/DIV2K_valid_LR_x2'
        self.dirname_GT = path_gt
        self.dirname_LQ = path_lq
        self.filelist_GT, self.filelist_LQ = [],[]
        for a,b,c in os.walk(self.dirname_GT):
            for _c in c:
                if '.png' in _c:
                    self.filelist_GT.append(os.path.join(a,_c))
        for a,b,c in os.walk(self.dirname_LQ):
            for _c in c:
                if '.png' in _c:
                    self.filelist_LQ.append(os.path.join(a,_c))
        self.filelist_GT = sorted(self.filelist_GT)
        self.filelist_LQ = sorted(self.filelist_LQ)
#         self.filelist_GT = sorted(os.listdir(self.dirname_GT))
#         self.filelist_LQ = sorted(os.listdir(self.dirname_LQ))
        self.crop_mode = crop_mode
        self.patch_size = patch_size
        self.scale = scale

    def __crop(self,img, pos, size):
        ow, oh = img.shape[1], img.shape[2]
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            img = img[:, x1:x1+size, y1:y1+size]

        return img

    def get_params(self, size, patch_size):
        w, h = size[1], size[2]
        new_h = h
        new_w = w

        x = random.randint(0, np.maximum(0, new_w - patch_size))
        y = random.randint(0, np.maximum(0, new_h - patch_size))

        return x, y
        
    def __len__(self):
        return len(self.filelist_GT)
        
    def __getitem__(self,idx):
        # data_ratio = len(self.filelist_LQ) / len(self.filelist_GT)
        
        img_name_GT = self.filelist_GT[idx]
        img_GT = cv2.imread(os.path.join(self.dirname_GT, img_name_GT), cv2.IMREAD_COLOR)
        img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
        img_GT = np.array(img_GT).astype('float32') / 255

        img_name_LQ = self.filelist_LQ[idx]
        img_LQ = cv2.imread(os.path.join(self.dirname_LQ, img_name_LQ))
        img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGR2RGB)
        # if not img_GT.shape == img_LQ.shape: #lr-lr인데 사이즈 미스매칭일때
#         img_LQ = cv2.resize(img_LQ,(img_GT.shape[1], img_GT.shape[0]))
        img_LQ = np.array(img_LQ).astype('float32') / 255
        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}

        if self.transform:
            sample = self.transform(sample)
            
        img_LQ = sample['img_LQ'].squeeze()
        img_GT = sample['img_GT'].squeeze()
        
        if self.crop_mode:
            pos = self.get_params(img_GT.shape, patch_size=self.patch_size*self.scale)
            gt_pos = None
            lq_pos= [floor(pos[0]/self.scale), floor(pos[1]/self.scale)] #hr-lr eval할때쓰기
            sample['img_LQ'] = self.__crop(img_LQ , lq_pos, self.patch_size)
            lqshape, hqshape = img_LQ.shape, img_GT.shape
            sample['img_GT'] = self.__crop(img_GT, pos, self.patch_size*self.scale)
        path = {'img_LQ': img_name_LQ, 'img_GT': img_name_GT}
#         print("-------------------------------")
#         print(sample['img_LQ'].shape, sample['img_GT'].shape)
#         print(img_name_GT, img_name_LQ)
#         print("-------------------------------")
#         assert(sample['img_LQ'].shape == sample['img_GT'].shape, f'IMAGE SIZE IS DIFFERENT ! {img_name_GT} & {img_name_LQ}')

        return sample, path

class LQGTDataset(Dataset):
    def __init__(self, lrpath,gtpath, transform, patch_size = 64, shuffle_mode=False, scale=2):
        super(LQGTDataset, self).__init__()
        self.transform = transform
        self.shuffle_mode = shuffle_mode
        self.dirname_GT = gtpath
        self.dirname_LQ = lrpath
        
        self.filelist_GT = os.listdir(self.dirname_GT)
        self.filelist_LQ = os.listdir(self.dirname_LQ)
        self.scale = scale
        if shuffle_mode :
            random.shuffle(self.filelist_GT)
        else: self.filelist_GT = sorted(self.filelist_GT)
        self.filelist_LQ = sorted(self.filelist_LQ)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.filelist_GT)

    def __getitem__(self,idx):
        # data_ratio = len(self.filelist_LQ) / len(self.filelist_GT)
        img_name_GT = self.filelist_GT[idx]
        img_GT = cv2.imread(os.path.join(self.dirname_GT, img_name_GT), cv2.IMREAD_COLOR)
        img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
        img_GT = np.array(img_GT).astype('float32') / 255

        img_name_LQ = self.filelist_LQ[idx]
        img_LQ = cv2.imread(os.path.join(self.dirname_LQ, img_name_LQ))
        img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGR2RGB)
        img_LQ = np.array(img_LQ).astype('float32') / 255

        # img_LQ: H x W x C (numpy array)
        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}
        sample['img_GT'], sample['img_LQ'] = self.__augment([sample['img_GT'], sample['img_LQ']], rotation=True, hflip=True)

        if self.transform:
            sample = self.transform(sample)
        pos = self.get_params(sample['img_GT'].shape, patch_size=self.patch_size*self.scale)
        gt_pos = None
        if self.shuffle_mode:
            lq_pos = self.get_params(sample['img_LQ'].shape, patch_size=self.patch_size)
        else:
            lq_pos= [ceil(pos[0]/self.scale), ceil(pos[1]/self.scale)] #hr-lr eval할때쓰기
        sample['img_LQ'] = self.__crop(sample['img_LQ'] , lq_pos, self.patch_size)
        lqshape, hqshape = sample['img_LQ'].shape, sample['img_GT'].shape
        sample['img_GT'] = self.__crop(sample['img_GT'], pos, self.patch_size*self.scale)
        path = {'img_LQ': img_name_LQ, 'img_GT': img_name_GT}
        return sample, path

    def __crop(self,img, pos, size):
        ow, oh = img.shape[1], img.shape[2]
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            img = img[:, x1:x1+size, y1:y1+size]

        return img


    def __augment(self, imgs, hflip=True, rotation=True, flows=None, return_status=False):
        """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

        We use vertical flip and transpose for rotation implementation.
        All the images in the list use the same augmentation.

        Args:
            imgs (list[ndarray] | ndarray): Images to be augmented. If the input
                is an ndarray, it will be transformed to a list.
            hflip (bool): Horizontal flip. Default: True.
            rotation (bool): Ratotation. Default: True.
            flows (list[ndarray]: Flows to be augmented. If the input is an
                ndarray, it will be transformed to a list.
                Dimension is (h, w, 2). Default: None.
            return_status (bool): Return the status of flip and rotation.
                Default: False.

        Returns:
            list[ndarray] | ndarray: Augmented images and flows. If returned
                results only have one element, just return ndarray.

        """
        hflip = hflip and random.random() < 0.5
        vflip = rotation and random.random() < 0.5
        rot90 = rotation and random.random() < 0.5

        def _augment(img):
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        def _augment_flow(flow):
            if hflip:  # horizontal
                cv2.flip(flow, 1, flow)
                flow[:, :, 0] *= -1
            if vflip:  # vertical
                cv2.flip(flow, 0, flow)
                flow[:, :, 1] *= -1
            if rot90:
                flow = flow.transpose(1, 0, 2)
                flow = flow[:, :, [1, 0]]
            return flow

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]

        if flows is not None:
            if not isinstance(flows, list):
                flows = [flows]
            flows = [_augment_flow(flow) for flow in flows]
            if len(flows) == 1:
                flows = flows[0]
            return imgs, flows
        else:
            if return_status:
                return imgs, (hflip, vflip, rot90)
            else:
                return imgs


    def get_params(self, size,patch_size):
        w, h = size[1], size[2]
        new_h = h
        new_w = w

        x = random.randint(0, np.maximum(0, new_w - patch_size))
        y = random.randint(0, np.maximum(0, new_h - patch_size))


        return x, y
