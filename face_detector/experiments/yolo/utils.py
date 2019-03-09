import torch
import cv2
from matplotlib import pyplot as plt

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

# def transform_bboxes()

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
        Predictions after objectness filtering and non-max supression.
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


def show_predictions(image_path, predictions):
    '''
    Displays predictions on the image provided in image_path.
    
    Arguments
    ---------
    image_path: str
        A path to an image.
    predictions: torch.FloatTensor
        Predictions after objectness filtering and non-max supression.
    
    todo:
    '''
    top_left_x = predictions[:, 0] - predictions[:, 2]/2
    top_left_y = predictions[:, 1] - predictions[:, 3]/2
    bottom_right_x = predictions[:, 0] + predictions[:, 2]/2
    bottom_right_y = predictions[:, 1] + predictions[:, 3]/2

    top_left_x = top_left_x.detach().int().numpy()
    top_left_y = top_left_y.detach().int().numpy()
    bottom_right_x = bottom_right_x.detach().int().numpy()
    bottom_right_y = bottom_right_y.detach().int().numpy()

    plt.figure(figsize=(7, 7))
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('make a proper resize for an image')
    img = cv2.resize(img, (416, 416))
    
    for i in range(len(top_left_x)):
        cv2.rectangle(img, (top_left_x[i], top_left_y[i]), (bottom_right_x[i], bottom_right_y[i]), (0, 0, 255), 2)
        cv2.putText(img, str(predictions[i, 6].detach().int().numpy()), (top_left_x[i], top_left_y[i] + 2 + 4), 
                    cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

    plt.imshow(img)
    plt.show()