import torch

def parse_cfg(file):
    '''
    Parses the original cfg file
    
    Argument
    --------
    file: str
        A path to cfg file.
        
    Output
    ------
    layers: list
        A list of dicts with config for each layer. 
        Note: the 0th element of the list contain config for network itself
    '''
    
    layers = []
    layer = {}

    with open(file, 'r') as readf:
        lines = readf.read().split('\n')
        # skip commented lines
        lines = [line for line in lines if not line.startswith('#')]
        # skip empty lines
        lines = [line for line in lines if not len(line) == 0]
        # remove all whitespaces
        lines = [line.replace(' ', '') for line in lines]

        for line in lines:

            # if the name of the layer (they are of form : [*])
            if line.startswith('[') and line.endswith(']'):

                # save the prev. layer as the next lines contains info for the next layer
                if len(layer) > 0:
                    layers.append(layer)
                    layer = {}

                # add the layer's name/type
                layer['name'] = line.replace('[', '').replace(']', '')

            # if not the name then parse agruments
            else:
                # all arguments follows the pattern: 'key=value'
                key, value = line.split('=')
                # add info to the layer
                layer[key] = value

        # append the last layer 
        layers.append(layer)
        
    return layers

def iou_vectorized(bboxes1, bboxes2):
    '''
    Calculates intersection over union between every bbox in bboxes1 with
    every bbox in bboxes2, i.e. Cartesian product of both sets.
    
    Arguments
    ---------
    bboxes1: torch.FloatTensor
        (M, 4) shapped tensor with M bboxes with 4 bbox coordinates (cx, cy, w, h).
    bboxes2: torch.FloatTensor
        (N, 4) shapped tensor with M bboxes with 4 bbox coordinates (cx, cy, w, h).
        
    Output
    ------
    : torch.FloatTensor
        (M, N) shapped tensor with (i, j) corresponding to IoU between i-th bbox 
        from bboxes1 with j-th bbox from bboxes2.
    '''
    M, D = bboxes1.shape
    N, D = bboxes2.shape
    
    # Transform coords of the 1st bboxes (y=0 is at the top, and increases downwards)
    top_left_x1 = bboxes1[:, 0] - bboxes1[:, 2]/2
    top_left_y1 = bboxes1[:, 1] - bboxes1[:, 3]/2
    bottom_right_x1 = bboxes1[:, 0] + bboxes1[:, 2]/2
    bottom_right_y1 = bboxes1[:, 1] + bboxes1[:, 3]/2
    # Transform coords of the 2nd bboxes
    top_left_x2 = bboxes2[:, 0] - bboxes2[:, 2]/2
    top_left_y2 = bboxes2[:, 1] - bboxes2[:, 3]/2
    bottom_right_x2 = bboxes2[:, 0] + bboxes2[:, 2]/2
    bottom_right_y2 = bboxes2[:, 1] + bboxes2[:, 3]/2

    # broadcasting 1st bboxes
    top_left_x1 = top_left_x1.view(M, 1)
    top_left_y1 = top_left_y1.view(M, 1)
    bottom_right_x1 = bottom_right_x1.view(M, 1)
    bottom_right_y1 = bottom_right_y1.view(M, 1)
    # broadcasting 2nd bboxes
    top_left_x2 = top_left_x2.view(1, N)
    top_left_y2 = top_left_y2.view(1, N)
    bottom_right_x2 = bottom_right_x2.view(1, N)
    bottom_right_y2 = bottom_right_y2.view(1, N)
    
    # calculate coords for intersection
    inner_top_left_x = torch.max(top_left_x1, top_left_x2)
    inner_top_left_y = torch.max(top_left_y1, top_left_y2)
    inner_bottom_right_x = torch.min(bottom_right_x1, bottom_right_x2)
    inner_bottom_right_y = torch.min(bottom_right_y1, bottom_right_y2)
    
    # area = side_a * side_b
    # clamp(x, min=0) = max(x, 0)
    # we make sure that the area is 0 if size of a side is negative
    # which means that inner_top_left_x > inner_bottom_right_x which is not feasible
    a = torch.clamp(inner_bottom_right_x - inner_top_left_x, min=0)
    b = torch.clamp(inner_bottom_right_y - inner_top_left_y, min=0)
    inner_area = a * b

    # finally we calculate union for each pair of bboxes
    out_area1 = (bottom_right_x1 - top_left_x1) * (bottom_right_y1 - top_left_y1)
    out_area2 = (bottom_right_x2 - top_left_x2) * (bottom_right_y2 - top_left_y2)
    out_area = out_area1 + out_area2 - inner_area

    return inner_area / out_area