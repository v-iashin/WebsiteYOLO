import os
import json
from tqdm import tqdm

import cv2

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
    
    def __init__(self, json_path, phase, model_in_width, transforms=None):
        
        with open(json_path, 'r') as fread:
            self.meta = json.load(fread)
            
        self.meta = {int(key): val for key, val in self.meta.items()}
        self.phase = phase
        self.model_in_width = model_in_width
        self.transforms = transforms
        
    def __getitem__(self, index):
        
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
        H_new, W_new, scale = scale_numbers(H, W, self.model_in_width)
        img = cv2.resize(img, (W_new, H_new))
        img, (pad_top, pad_bottom, pad_left, pad_right) = letterbox_pad(img)
        
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
            gt_bboxes[:, :4] = gt_bboxes[:, :4] / self.model_in_width
            
            # batch index and face class number are going to be 0
            two_columns_with_zeros = torch.zeros(len(gt_bboxes), 2)
            print(two_columns_with_zeros.shape, gt_bboxes.shape)
            targets = torch.cat([two_columns_with_zeros, gt_bboxes], dim=1)
            
            return img, targets
        
        elif self.phase == 'test':
    
            return img, None

    def __len__(self):
        return len(self.meta)

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
