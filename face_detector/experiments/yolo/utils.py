import torch
import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm as colormap
from matplotlib.colors import Normalize

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

def get_corner_coords(bboxes):
    '''
    Transforms the bounding boxes coordinate from (cx, cy, w, h) into
    (top_left_x, top_left_y, bottom_right_x, bottom_right_y), 
    i.e. into corner coordinates.
    
    Argument
    --------
    bboxes: torch.FloatTensor
        A tensor of size (P, D) where D should contain info about the coords
        in the following order (center_x, center_y, width, height). Note:
        D can be higher than 4.
    
    Outputs
    -------
    top_left_x, top_left_y, bottom_right_x, bottom_right_y: torch.FloatTensors
        Transformed coordinates for bboxes: top-left corner coordinates for x and y
        and bottom-right coordinates for x and y respectively.
    '''
    top_left_x = bboxes[:, 0] - bboxes[:, 2]/2
    top_left_y = bboxes[:, 1] - bboxes[:, 3]/2
    bottom_right_x = bboxes[:, 0] + bboxes[:, 2]/2
    bottom_right_y = bboxes[:, 1] + bboxes[:, 3]/2

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

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
    top_left_x1, top_left_y1, bottom_right_x1, bottom_right_y1 = get_corner_coords(bboxes1)
    # Transform coords of the 2nd bboxes
    top_left_x2, top_left_y2, bottom_right_x2, bottom_right_y2 = get_corner_coords(bboxes2)

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

def objectness_filter_and_nms(predictions, classes, obj_thresh=0.8, nms_thresh=0.4):
    '''
    Performs filtering according objectness score and non-maximum supression on predictions.
    
    Arguments
    ---------
    predictions: torch.FloatTensor
        A tensor of size (B, P, 5+classes) with predictions.
        B -- batch size; P -- number of predictions for an image, 
        i.e. 3 scales and 3 anchor boxes and
        For example: P = (13*13 + 26*26 + 52*52) * 3 = 10647;
        5 + classes -- (cx, cy, w, h, obj_score, {prob_class}).
    classes: int
        An integer with the number of classes to detect.
    obj_thresh: float
        A float that corresponds to the lowest objectness score the detector allows.
    nms_thresh: float
        Corresponds to the highest IoU the detector allows.
        
    Output
    ------
    predictions: torch.FloatTensor
        Predictions after objectness filtering and non-max supression (same size
        as predictions in arguments but with a different P).
    '''
    
    # iterate for images in a batch
    for i, prediction in enumerate(predictions):
        ## objectness thresholding
        
        # If prediction's (bbox') score is higher than obj_thress keep the prediction
        # the fourth (fifth) element is objectness score
        objectness_mask = (prediction[:, 4] > obj_thresh)
        prediction = prediction[objectness_mask]

        # if no object on an image found, continue with the next image
        if prediction.size(0) == 0:
            continue
        
        ## non-max supression
        # The idea is as follows. If a prediction "survived" objectness filtering
        # then it is considered meaningful. Since we may have multiple detections of 
        # one object on an image we need to filter out those predictions that have
        # substantial (more than nms_thresh) overlap, or IoU, with the box with highest
        # class score. Also note that one image might contact more than object of the same class.
        # So, as they don't have high IoU with the box with highest class score, they will be kept
        # in the list of predictions
        
        # for each prediction we save the class with the maximum class score
        pred_score, pred_classes = torch.max(prediction[:, 5:5+classes], dim=1, keepdim=True)
        # class scores are replaced by the highest class score and corresponding class
        # detections: (cx, cy, w, h, obj_score, top_class_score, top_class_idx)
        prediction = torch.cat((prediction[:, :5], pred_score.float(), pred_classes.float()), dim=1)
        # we are going to iterate through classes, so, first, we select the set of unique classes
        unique_classes = pred_classes.unique().float()
        
        # initialize the list of filtered detections
        detections_after_nms = []

        for cls in unique_classes:
            # select only the entries for a specific class
            prediction_4_cls = prediction[prediction[:, 6] == cls]
            # then we sort predictions for a specific class by objectness score (high -> low)
            sort_pred_idxs = torch.sort(prediction_4_cls[:, 4], descending=True)[1]
            prediction_4_cls = prediction_4_cls[sort_pred_idxs]      
            
            # next we want to fill out detections_after_nms with only with those objects
            # that has a unique position, i.e. low IoU with other predictions.
            # The idea here is to append (save) the first prediction in the prediction_4_cls
            # and calculate IoUs with the rest predictions in that prediction_4_cls of the 
            # ordered list. Next, the predictions with the high IoU
            # with the first prediction in prediction_4_cls will be discarded. 
            # For the next iteration, the first prediction will be the prediction with
            # the highest obj score among the ones that are left.
            # exit the loop when there is no prediction left after the nms
            while len(prediction_4_cls) > 0:
                # we append the first prediction for a specific class to the list of predictions.
                # We can do this because we ordered the prediction_4_cls beforehand.
                detections_after_nms.append(prediction_4_cls[0].unsqueeze(0))
                
                # also stop when this is the last prediction in prediction_4_cls
                if len(prediction_4_cls) == 1:
                    break
                
                # calculate IoUs with the first pred in prediction_4_cls and the rest of them
                ious = iou_vectorized(prediction_4_cls[0, :5].unsqueeze(0), prediction_4_cls[1:, :5])
                # when iou_vectorized inputs two tensors, the ious.shape is (N, M) but now N = 1
                # and [ious < nms_thresh] should be one dimesional
                ious = ious.reshape(-1)
                # filter out the first prediction (1:) and the ones with high IoU with the 0th pred
                prediction_4_cls = prediction_4_cls[1:][ious < nms_thresh]
        
        # as detections_after_nms is a list, we concatenate its elements to a tensor
        predictions = torch.cat(detections_after_nms)

    return predictions

def scale_numbers(num1, num2, largest_num_target):
    '''
    Scales two numbers (for example, dimensions) keeping aspect ratio.
    
    Arguments
    ---------
    num1: float or int
        The 1st number (dim1).
    num2: float or int
        The 2nd number (dim2).
    largest_num_target: int
        The expected size of the largest number among 1st and 2nd numbers.
        
    Outputs
    -------
    : (int, int, float)
        Two scaled numbers such that the largest is equal to largest_num_target 
        maintaining the same aspect ratio as num1 and num2 in input. Also,
        returns a scalling coefficient.
        Note: two ints are returned.
        
    Examples
    --------
        scale_numbers(832, 832, 416) -> (416, 416, 0.5)
        scale_numbers(223, 111, 416) -> (416, 207, 1.8654708520179373)
        scale_numbers(100, 200, 416) -> (208, 416, 2.08)
        scale_numbers(200, 832, 416) -> (100, 416, 0.5)
    '''
    # make sure the arguments are of correct types
    assert isinstance(largest_num_target, int), 'largest_num_target should be "int"'
    
    # to make the largest number to be equal largest_num_target keeping aspect ratio
    # we need, first, to estimate by how much the largest number is smaller (larger) 
    # than largest_num_target and, second, to scale both numbers by this ratio.
    
    # select the maximum among two numbers
    max_num = max(num1, num2)
    # calculate scalling coefficient
    scale_coeff = largest_num_target / max_num
    # scale both numbers
    num1 = num1 * scale_coeff
    num2 = num2 * scale_coeff
    
    # making sure that the numbers has int type
    return int(num1), int(num2), scale_coeff

def letterbox_pad(img, net_input_size, color=(127.5, 127.5, 127.5)):
    '''
    Adds padding to an image according to the original implementation of darknet.
    Specifically, it will pad the image up to (net_input_size x net_input_size) size.
    
    Arguments
    ---------
    img: numpy.ndarray
        An image to pad.
    net_input_size: int
        The network's input size.
    color: (float or int, float or int, float or int) \in [0, 255]
        The RGB intensities. The image will be padded with this color.
        
    Output
    ------
    img: numpy.ndarray
        The padded image.
    pad_sizes: (int, int, int, int)
        The sizes of paddings. Used in show_prediction module where we need to shift
        predictions by the size of the padding.
    '''
    # make sure the arguments are of correct types
    assert isinstance(img, np.ndarray), '"img" should have numpy.ndarray type'
    assert isinstance(net_input_size, int), '"net_input_size" should have int type'
    assert (
        isinstance(color[0], (int, float)) and 
        isinstance(color[1], (int, float)) and 
        isinstance(color[2], (int, float))
    ), 'each element of "color" should contain either a float or an int'
    
    H, W, C = img.shape
    max_side_len = max(H, W)
    
    # if width is higher than height then, to make a squared-shaped image, we need
    # to pad the height; else, we need to pad width.
    if W > H:
        # calculates how much should be padded "on top" which is a half of 
        # the difference between the target size and the current height
        pad_top = (max_side_len - H) // 2
        # another half is added to the bottom
        pad_bottom = max_side_len - (H + pad_top)
        pad_left = 0
        pad_right = 0
        
    else:
        pad_top = 0
        pad_bottom = 0
        # calculates how much should be padded "on left" which is a half of 
        # the difference between the target size and the current width
        pad_left = (max_side_len - W) // 2
        pad_right = max_side_len - (W + pad_left)
    
    # pad_widths should contain three pairs (because of 3d) of padding sizes:
    # first pair adds rows [from top and bottom], 
    # second adds columns [from left to right],
    # the third adds nothing because we pad only spatially, not channel-wise
    pad_widths = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    # for each padding we specify a color (constant parameter)
    color = [[color, color], [color, color], [0, 0]]
    # perform padding
    img = np.pad(img, pad_widths, 'constant', constant_values=color)
    # save padding sizes
    pad_sizes = (pad_top, pad_bottom, pad_left, pad_right)
    
    return img, pad_sizes

def show_predictions(image_path, predictions, classes, net_input_size):
    '''
    Shows the predictions on the image provided in image_path.
    
    Arguments
    ---------
    image_path: str
        A path to an image.
    predictions: torch.FloatTensor
        Predictions after letterbox padding, objectness filtering and non-max supression.
        A tensor of size (P, 4+1+2) -- without 'batch'-dimension; includes bbox attributes,
        objectess score, top class score, and top class index.
        Note: predictions tensor is not limited to has 7 columns -- it can be more but not less.
        
    # TODO other args
    '''
    # make sure the arguments are of correct types
    assert isinstance(image_path, str), '"image_path" should be str'
    assert isinstance(predictions, (torch.FloatTensor, torch.cuda.FloatTensor)), \
            '"predictions" should be either torch.FloatTensor or torch.cuda.FloatTensor'
    
    # parameters of the vizualization
    figsize = (7, 7)
    
    # First, we transform coordinates (cx, cy, w, h, obj_score, {prob_class}) to
    # corner coordinates: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_corner_coords(predictions)
    
    # detach values from the computation graph, take the int part and transform to numpy arrays
    top_left_x = top_left_x.detach().int().numpy()
    top_left_y = top_left_y.detach().int().numpy()
    bottom_right_x = bottom_right_x.detach().int().numpy()
    bottom_right_y = bottom_right_y.detach().int().numpy()

    # initialize the figure environment
    plt.figure(figsize=figsize)
    # read an image and transform the colors from BGR to RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # since predictions are made for a resized and letterbox padded images, 
    # the bounding boxes have to be scaled and shifted accordingly
    # for that we again ..........
    H, W, C = img.shape
    H_new, W_new, scale = scale_numbers(H, W, net_input_size)
#     _, pad_sizes
    img = cv2.resize(img, (W_new, H_new))
    img, _ = letterbox_pad(img, net_input_size, color=(128, 128, 128))
    print('make bboxes on image more pleasant')
    
    # add each prediction on the image and captures it with a class number
    for i in range(len(predictions)):
        # we need to scale and shift corner coordinates because we used the letterbox padding
        top_left_coords = (top_left_x[i], top_left_y[i])
        bottom_right_coords = (bottom_right_x[i], bottom_right_y[i])
        # predicted class
        class_int = predictions[i, 6].detach().int().numpy()
        # class_name = predictions[i, 6].detach().int().numpy()
        # color of bbox
        
        
        # add a bbox
        cv2.rectangle(img, top_left_coords, bottom_right_coords, (128, 128, 128), 2)
        # adds the class number
        cv2.putText(img, str(class_int), (top_left_x[i], top_left_y[i] + 2 + 4), 
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    plt.imshow(img)
    plt.show()
    
    
def predict_and_save(img_path, save_path, model, device, labels_path='./data/coco.names', show=False):
    '''
    Predicts objects on an image, draws the bounding boxes around the predicted objects,
    and saves the image.
    
    TODO check grammar
    
    Arguments
    ---------
    image_path: str
        The path to an image with objects to predict.
    save_path: str
        The path for the output image with detected objects.
    model: Darknet
        The model which will be used for inference.
    device: torch.device or str
        Device for calculations.
    labels_path: str
        The path to the object names.
    show: bool
        Whether to show the output image with bounding boxes, for example, in jupyter notebook
    '''
    # make sure the arguments are of correct types
    assert isinstance(img_path, str), '"img_path" should be str'
    assert save_path is None or isinstance(save_path, str), 'save_path should be NoneType or str'
    assert isinstance(labels_path, str), '"labels_path" should be str'
#     assert isinstance(model, darknet.Darknet), 'model should be a Darknet module'
    assert isinstance(device, (torch.device, str)), 'device should be either torch.device or str'
    assert isinstance(show, bool), 'show should be boolean'

    # parameters of the vizualization: color palette, figsize to show, 
    # label parameters, jpeg quality
    norm = Normalize(vmin=0, vmax=model.classes)
    color_map = colormap.tab10
    figsize = (15, 15)
    line_thickness = 2
    font_face = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.1
    font_color = [255, 255, 255] # white
    font_thickness = 1
    jpg_quality = 80

    # make a dict: {class_number: class_name}
    names = [name.replace('\n', '') for name in open(labels_path, 'r').readlines()]
    num2name = {num: name for num, name in enumerate(names)}

    # read an image and transform the colors from BGR to RGB
    img_raw = cv2.imread(img_path)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # add letterbox padding and save the pad sizes and scalling coefficient
    # to use it latter when drawing bboxes on the original image
    H, W, C = img_raw.shape
    H_new, W_new, scale = scale_numbers(H, W, model.in_width)
    img = cv2.resize(img_raw, (W_new, H_new))
    img, pad_sizes = letterbox_pad(img, model.in_width)

    # HWC -> CHW, scale intensities to [0, 1], send to pytorch, add 'batch-'dimension
    img = img.transpose((2, 0, 1))
    img = img / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    # make prediction and apply objectness filtering and nms
    predictions = model(img, device)
    predictions = objectness_filter_and_nms(predictions, model.classes)

    # since the predictions are made for a resized and padded images, 
    # the bounding boxes have to be scaled and shifted back
    # for that, we shift and scale back the bboxes' attributes
    pad_top, pad_bottom, pad_left, pad_right = pad_sizes
    predictions[:, 0] = (predictions[:, 0] - pad_left) / scale
    predictions[:, 1] = (predictions[:, 1] - pad_top) / scale
    predictions[:, 2] = predictions[:, 2] / scale
    predictions[:, 3] = predictions[:, 3] / scale

    # the, transform the coordinates (cx, cy, w, h) into corner coordinates: 
    # (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_corner_coords(predictions)

    # detach values from the computation graph, take the int part and transform to np.ndarray
    top_left_x = top_left_x.detach().int().numpy()
    top_left_y = top_left_y.detach().int().numpy()
    bottom_right_x = bottom_right_x.detach().int().numpy()
    bottom_right_y = bottom_right_y.detach().int().numpy()

    # if show initialize a figure environment
    if show:
        plt.figure(figsize=figsize)

    # add each prediction on the image and captures it with a class number
    for i in range(len(predictions)):

        ## ADD BBOXES
        # we need to scale and shift corner coordinates because we used the letterbox padding
        top_left_coords = (top_left_x[i], top_left_y[i])
        bottom_right_coords = (bottom_right_x[i], bottom_right_y[i])
        # predicted class number
        class_int = int(predictions[i, 6].detach().numpy())
        # select the color for a class according to its label number and scale it to [0, 255]
        bbox_color = color_map(class_int)[:3]
        bbox_color = list(map(lambda x: x * 255, bbox_color))
        # add a bbox
        cv2.rectangle(img_raw, top_left_coords, bottom_right_coords, bbox_color, line_thickness)

        ## ADD A LABLE FOR EACH BBOX INSIDE THE RECTANGLE WITH THE SAME COLOR AS THE BBOX ITSELF
        # predicted class name to put on a bbox
        class_name = num2name[class_int]
        # size for the text
        text_size = cv2.getTextSize(class_name, font_face, font_scale, font_thickness)[0]
        # bottom right coordinates for the small rectangle for the label
        bottom_right_coords_ = top_left_coords[0] + text_size[0] + 4, top_left_coords[1] + text_size[1] + 4
        # adds a small rectangle of the same color to be the background for the label
        cv2.rectangle(img_raw, top_left_coords, bottom_right_coords_, bbox_color, cv2.FILLED)
        # position for text
        xy_position = (top_left_x[i] + 2, top_left_y[i] + text_size[1])
        # adds the class label
        cv2.putText(img_raw, class_name, xy_position, font_face, font_scale, font_color, font_thickness)

    # if show, then, show and close the environment
    if show:
        plt.imshow(img_raw)

    # RGB -> BGR and save output image
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_raw, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    