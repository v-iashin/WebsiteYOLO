import os
import json
from tqdm import tqdm

import cv2
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data.dataset import Dataset

from utils import scale_numbers, letterbox_pad

'''
The dataset folder is expected to have
    the following structure:
    
    ./Submission_example/
        0--Parade/
            0_Parade_marchingband_1_20.txt
    ./wider_face_split/
        wider_face_test.mat
        wider_face_train.mat
        wider_face_test_filelist.txt
        wider_face_val.mat
        wider_face_train_bbx_gt.txt
        readme.txt
        wider_face_val_bbx_gt.txt
    ./WIDER_train/
        images/
            0--Parade/
                0_Parade_marchingband_1_100.jpg
                ...
            1--Handshaking/
                1_Handshaking_Handshaking_1_102.jpg
                ...
            ...
    ./WIDER_val/
        (similar to ./WIDER_train/)
    ./WIDER_test/
        (similar to ./WIDER_train/)
'''

class WIDERdataset(Dataset):
    '''
    TODO: doc
    '''
    
    def __init__(self, json_path, phase, model_width, transforms=None):
        
        with open(json_path, 'r') as fread:
            self.meta = json.load(fread)
            
        self.meta = {int(key): val for key, val in self.meta.items()}
        self.phase = phase
        self.model_width = model_width
        self.transforms = transforms
        
    def __getitem__(self, index, show_examples=False):
        
        if self.phase in ['train', 'val']:
            full_file_path, size_HW, gt_bboxes = self.meta[index].values()
            
        elif self.phase == 'test':
            full_file_path, size_HW = self.meta[index].values()

        # read image and invert colors BGR -> RGB
        img = cv2.imread(full_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # add letterbox padding and save the pad sizes and scalling coefficient
        # to use it to shift the bboxes' coordinates
        H, W = size_HW
        H_new, W_new, scale = scale_numbers(H, W, self.model_width)
        img = cv2.resize(img, (W_new, H_new))
        img, (pad_top, pad_bottom, pad_left, pad_right) = letterbox_pad(img)
        
        # applying transforms for an img (to be reconsidered for augmentation)
        if self.transforms:
            img = self.transforms(img)
        
        if self.phase in ['train', 'val']:
            # top_left_x, top_left_y, w, h, blur, expr, illum, inv, occl, pose
            gt_bboxes = torch.tensor(gt_bboxes).float()
            # transforms corner bbox coords into center coords
            gt_bboxes[:, 0] = gt_bboxes[:, 0] + gt_bboxes[:, 2] // 2
            gt_bboxes[:, 1] = gt_bboxes[:, 1] + gt_bboxes[:, 3] // 2
            
            # scale all bbox coordinates and dimensions to fit resizing
            gt_bboxes[:, :4] = gt_bboxes[:, :4] * scale

            # shift center coordinates (x and y) to fit letterbox padding
            gt_bboxes[:, 0] = gt_bboxes[:, 0] + pad_left
            gt_bboxes[:, 1] = gt_bboxes[:, 1] + pad_top

            # scale bbox coordinates and dimensions to [0, 1]
            gt_bboxes[:, :4] = gt_bboxes[:, :4] / self.model_width
            
            # batch index and face class number are going to be 0
            two_columns_with_zeros = torch.zeros(len(gt_bboxes), 2)
            targets = torch.cat([two_columns_with_zeros, gt_bboxes], dim=1)
            
            if show_examples:
                self.show_examples(img, targets)
            
            return img, targets
        
        elif self.phase == 'test':
    
            return img, None
    
    def collate_wider(self, batch):
        '''
        TODO
        '''
        # add image index to the targets tensor because we need to know
        # to which image in the batch a g.t. object correspond to as
        # we are going to contatenate them in one tensor (see later here
        # and also in make_targets in darknet for the loss calculation)
        for img_idx, (image, targets) in enumerate(batch):
            targets[:, 0] = img_idx

        # extract images and targets from tuples
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # images is a list of tensors B x (C, H, W) -> (B, C, H, W) tensor
        # but first the batch dim is added
        images = [img.unsqueeze(0) for img in images]
        images = torch.cat(images, dim=0)

        # targets is a list of tensors, I will concat. them on the first dim
        targets = torch.cat(targets, dim=0)

        return images, targets
    
    def show_examples(self, img, targets, figsize=10):
        '''
        TODO
        '''
        if self.transforms:
            img = ToNumpy()(img)
                    
        H, W, C = img.shape
                    
        for target in targets:
            top_left_x = int(target[2] * W) - int((target[4] * W) // 2)
            top_left_y = int(target[3] * H) - int((target[5] * H) // 2)
            bottom_right_x = int(target[2] * W) + int((target[4] * W) // 2)
            bottom_right_y = int(target[3] * H) + int((target[5] * H) // 2)
            top_left_coords = top_left_x, top_left_y
            bottom_right_coords = bottom_right_x, bottom_right_y
            cv2.rectangle(img, top_left_coords, bottom_right_coords, (255, 255, 255), 2)
                
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.show()
    
    def __len__(self):
        
        return len(self.meta)
    
class ToTensor(object):
    """
    Convert ndarrays to Tensors.
    TODO:
    """
    
    def __call__(self, x):
        # making sure that the tensors are of a proper type
        x = x.astype(np.float32)
        
        # swap channel axis because
        # cv2 image: H x W x C
        # torch image: C x H x W
        x = x.transpose(2, 0, 1)
        
        # [0, 255] -> [0, 1]
        x = x / 255
        
        return torch.from_numpy(x)
    
class ToNumpy(object):
    """
    Convert to Tensors to ndarrays.
    TODO:
    """
    
    def __call__(self, x):
        # restores back the ToTensor transformation
        x = x * 255
        x = x.permute(1, 2, 0)
        x = x.int()
        
        return x.numpy()

def read_wider_meta(data_root_path, phase):
    '''
    Parses WIDER ground truth data.
        
    Argument
    --------
    data_root_path: str
        A path to the ground truth dataset. It is expected to have the '.txt' files
        extension.
        
    Output
    ------
    meta: dict
        A map between a training example number (index)and and another dict which 
        contains file path  image size (size_HW) of this file and a list of lists 
        containing ground truth bounding box coordinates (top_left_x, top_left_y, w, h) --
        ints and attributes (blur, expression, illumination, invalid, occlusion, pose) 
        (gt_bboxes) -- binary. For more information about the attributes see readme.txt.
        
    Dataset Head
    ------------
    train and validation:
        ...
        17--Ceremony/17_Ceremony_Ceremony_17_1048.jpg
        4
        25 421 39 46 1 0 0 0 0 0 
        129 400 34 40 1 0 0 0 0 0 
        334 139 93 136 0 0 0 0 0 0 
        467 95 108 147 0 0 0 0 0 0 
        17--Ceremony/17_Ceremony_Ceremony_17_444.jpg
        1
        332 263 341 461 0 0 0 0 0 0 
        17--Ceremony/17_Ceremony_Ceremony_17_452.jpg
        6
        221 378 13 26 2 0 0 0 0 0 
        312 369 20 29 2 0 0 0 0 0 
        309 336 14 20 2 0 0 0 0 0 
        843 386 12 22 2 0 0 0 0 0 
        932 391 16 22 1 0 0 0 0 0 
        976 390 16 24 2 0 1 0 0 0
        ...
        
    test:
        0--Parade/0_Parade_marchingband_1_737.jpg
        0--Parade/0_Parade_marchingband_1_494.jpg
        0--Parade/0_Parade_Parade_0_338.jpg
        0--Parade/0_Parade_marchingband_1_533.jpg
        0--Parade/0_Parade_marchingband_1_62.jpg
        0--Parade/0_Parade_marchingband_1_184.jpg
        0--Parade/0_Parade_marchingband_1_120.jpg
        ...
    '''
    # depending on the phase we change the paths    
    if phase in ['train', 'val']:
        meta_path = os.path.join(data_root_path, f'wider_face_split/wider_face_{phase}_bbx_gt.txt')
    
    elif phase == 'test':
        meta_path = os.path.join(data_root_path, f'wider_face_split/wider_face_test_filelist.txt')
        
    images_path = os.path.join(data_root_path, f'WIDER_{phase}/images')
    
    
    meta = {}
    # index for a training example
    idx = 0

    with open(meta_path, 'r') as rfile:
        
        # since the files is going to be read line-by-line we don't know
        # how many lines are there. Also, it's been decided not to use 
        # for-loop because it would add a lot of 'if's and 'continue' lines. 
        # Hence 'while True' loop.
        while True:
            # short_file_path is always followed by bbox_count in train and val
            # whereas in test set only file names are presented.
            
            # short_file_path is a path inside the wider dataset
            # Also, we remove carrige return in the line
            short_file_path = rfile.readline().replace('\n', '')
            
            # if the end of the file reached, return
            if short_file_path == '':
                rfile.close()
                return meta
            
            # join the path inside the dataset with the path to dataset
            full_file_path = os.path.join(images_path, short_file_path)
            
            # also, the size of an image is going to be stored
            H, W, C = cv2.imread(full_file_path).shape
            
            # file_info is going to contain image path, size and g.t. bboxes if available
            file_info = {
                'full_file_path': full_file_path, 
                'size_HW': (H, W)
            }
            
            
            # if it is a test file we don't have g.t. bbox info
            if phase == 'test':
                # add image size to meta
                meta[idx] = file_info
                idx += 1
                continue
            
            # bbox_count how many g.t. bboxes are going to be for this image
            # Also, we remove carrige return in the line
            bbox_count = int(rfile.readline().replace('\n', ''))
            
            # bboxes are going to be list of lists
            gt_bboxes = []
            
            # we are going to read the new lines and append bbox coordinates with 
            # their attributs to the gt_bboxes
            for _ in range(bbox_count):
                attributes = rfile.readline()
                attributes = attributes.replace('\n', '').split(' ')
                attributes = [int(att) for att in attributes if len(att) > 0]

                gt_bboxes.append(attributes)
            
            # add gt_boxes info to file_info and add this dict to the meta dict
            file_info['gt_bboxes'] = gt_bboxes
            meta[idx] = file_info
            idx += 1
