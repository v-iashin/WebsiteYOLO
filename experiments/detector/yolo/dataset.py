import os
import cv2

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

def read_wider_meta(data_root_path, phase):
    '''
    Parses WIDER ground truth data.
        
    Argument
    --------
    data_root_path: str
        A path to the ground truth dataset. It is expected to have the '.txt'
        extension.
        
    Output
    ------
    meta: dict
        A map between a file path and another dict that contain image size (size_HW)
        of this file and a list of lists containing ground truth bounding box 
        coordinates (top_left_x, top_left_y, w, h) and attributes (blur, expression, 
        illumination, invalid, occlusion, pose) (gt_bboxes). 
        For more information about the attributes see readme.txt.
        
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
    
    images_path = os.path.join(data_root_path, f'WIDER_{phase}/images')
    
    if phase in ['train', 'val']:
        meta_path = os.path.join(data_root_path, f'wider_face_split/wider_face_{phase}_bbx_gt.txt')
    
    elif phase == 'test':
        meta_path = os.path.join(data_root_path, f'wider_face_split/wider_face_test_filelist.txt')
    
    meta = {}

    with open(meta_path, 'r') as rfile:
        
        # since the files is going to be read line-by-line we don't know
        # how many lines are there. Hence 'while True' loop.
        while True:
            # short_file_path is always followed by bbox_count in train and val
            # whereas in test set only file names are presented.
            
            # short_file_path is a path inside the wider dataset
            # Also, we remove carrige return in the line
            short_file_path = rfile.readline().replace('\n', '')
            
            # if the end of the file reached, return
            if short_file_path == ''
                rfile.close()
                return meta
            
            # join the path inside the dataset with the path to dataset
            full_file_path = os.path.join(images_path, short_file_path)
            
            if phase == 'test':
                # also, the size of an image is going to be stored
                H, W, C = cv2.imread(full_file_path).shape
                
                meta[full_file_path] = {'size_HW': (H, W)}
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
            
            # make a dict with the size info and gg_boxes for this image
            # and add this info to the meta dict
            file_info = {'size_HW': (H, W), 'gt_bboxes': gt_bboxes}
            meta[full_file_path] = file_info
