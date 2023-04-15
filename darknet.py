import os

import numpy as np
import torch
from torch import nn

from utils import iou_vectorized, parse_cfg


class RouteLayer(nn.Module):
    '''Route layer outputs the concatenated outputs from the specified layers.'''

    def __init__(self, routes):
        '''
        Arguments
        ---------
        routes: list
            A list of indices that correspond to the previous layers. The outputs
            from these layers will be used as the output from the route layer.

            Examples:
            [-4]: the output of the route layer in this case is the
                output of the 4th layer before this route layer.
            [-1, 61]: the output of the route layer is the concatenation
                of the previous and 61th layers on depth ('channel') dimension.
        '''
        super(RouteLayer, self).__init__()
        self.routes = routes

class ShortcutLayer(nn.Module):
    '''Similarly to Routelayer shortcut layer functions as a dummy layer and only saves
    the index of the layer to use the shortcut from.'''

    def __init__(self, frm):
        '''
        Arguments
        ---------
        frm: int
            The index of the layer which will be used as the shotcut connection with the
            current layer. Think of it as a ResNet's shortcut connection. For more
            information please see the 'Examples' to RouteLayer.
        '''
        super(ShortcutLayer, self).__init__()
        self.frm = frm

class YOLOLayer(nn.Module):
    '''Similarly to the previous layers, YOLO layer in defined'''

    def __init__(self, anchors, classes, num, jitter, ignore_thresh, truth_thresh, random, model_width):
        '''
        Arguments
        ---------
        anchors: list
            A list of tuples of pairs of ints corresponding to initial sizes of
            bounding boxes (width and height). In YOLO v3 there are 3 pairs of ints.
        classes: int
            The number of classes. In COCO there are 80 classes.
        num: int
            The number of anchors. In YOLO v3 there are 9 anchors.
        jitter: float \in [0, 1]
            The parameter corresponds to the invariace of the random cropping during
            training. The larger, the higher the invariance in size and aspect ratio.
        ignore_thresh: float
            TODO (The parameter is used for debugging and not important)
        truth_thresh: float
            TODO(The parameter is used for debugging and not important)
        random: int (??)
            If 1 then YOLO will perform data augmentation to generalize model for resized
            images (performs resizing).
        model_width: int
            The width of a model specified in the config.
            `in_width = % 32` should be 0. Example: 416 or 608
        '''
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.num = num
        self.jitter = jitter
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.random = random
        self.model_width = model_width
        self.EPS = 1e-16
        # 100 and 1 are taken from github.com/eriklindernoren/PyTorch-YOLOv3
        self.noobj_coeff = 100
        self.obj_coeff = 1
        self.ignore_thresh = 0.5
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x, targets, device, input_width):
        '''
        Arguments
        ---------
        x: torch.FloatTensor
            An image of size (B, C, G, G).
        targets: None or torch.FloatTensor
            None if no targets are provided (for inference).
            If provided, ground truth (g.t.) bboxes and their classes.
            The tensor has the number of rows according to the number
            of g.t. bboxes in the whole batch and 6 columns:
                - the image idx within the batch;
                - the g.t. label corresponding to this bbox;
                - center coordinates x, y \in (0, 1);
                - bbox width and height
            Note: the coordinates account for letter padding which means
                  that the coordinates for the center are shifted accordingly
        device: torch.device
            The device to use for calculation: torch.device('cpu'), torch.device('cuda:?')

        Outputs
        -------
        predictions: torch.FloatTensor
            A tensor of size (B, P, 5+classes) with predictions.
            B -- batch size; P -- number of predictions for an image,
            i.e. 3 scales and 3 anchor boxes and
            For example: P = (13*13 + 26*26 + 52*52) * 3 = 10647;
            5 + classes -- (cx, cy, w, h, obj_score, {prob_class}).
        loss: float or int
            The accumulated loss (float) after passing through every
            YOLO layers at each scale. If targets is None, returns zero (int).
        '''
        # input size: (B, (4+1+classes)*num_achors=255, G_scale, G_scale)
        B, C, G, G = x.size()
        # read layer's info
        anchors_list = self.anchors
        classes = self.classes
        model_width = self.model_width
        num_anchs = len(anchors_list)
        # bbox coords + obj score + class scores
        num_feats = 4 + 1 + classes
        # we need to define it here to handle the case when targets is None
        loss = 0

        # transform the predictions
        # (B, ((4+1+classes)*num_achors), Gs, Gs)
        # -> (B, num_achors, w, h, (4+1+classes))
        x = x.view(B, num_anchs, num_feats, G, G)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        # Why do we need to calculate c_x and c_y?
        # So far, the predictions for center coordinates are just logits
        # that are expected to be mapped into (0, 1) by sigmoid.
        # After sigmoid, the values represent the position of the center
        # coordinates for an anchor in a respective cell. Specifically,
        # (1, 1) means the center of the anchor is in the bottom-right
        # corner of the respective cell. However, we would like to predict
        # the pixel position for the original image. For that, we add the
        # coordinates of x and y of the grid to each "sigmoided" value
        # which tells which position on the grid a prediction has.
        # To transform these grid coordinates to original image, we
        # multiply these values by the stride (=orig_size / cell_width).
        c_x = torch.arange(G).view(1, 1, 1, G).float().to(device)
        c_y = torch.arange(G).view(1, 1, G, 1).float().to(device)

        # Why we need to calculate p_wh?
        # YOLO predicts only the coefficient which is used to scale the bounding
        # box priors. Therefore, we need to calculate those priors: p_wh.
        # Note: yolo predicts the coefficient in log-scale. For this reason
        # we apply exp() on it.
        # stride = the size of the grid cell side
        stride = input_width // G
        # After dividing anchors by the stride, they represent the size size of
        # how many grid celts they are overlapping: 1.2 = 1 and 20% of a grid cell.
        # After multiplying them by the stride, the pixel values are going to be
        # obtained.
        anchors_list = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors_list]
        anchors_tensor = torch.tensor(anchors_list, device=device)
        # (A, 2) -> (1, A, 1, 1, 2) for broadcasting
        p_wh = anchors_tensor.view(1, num_anchs, 1, 1, 2)

        # prediction values for the *loss* calculation (training)
        pred_x = torch.sigmoid(x[:, :, :, :, 0])
        pred_y = torch.sigmoid(x[:, :, :, :, 1])
        pred_wh = x[:, :, :, :, 2:4]
        pred_obj = torch.sigmoid(x[:, :, :, :, 4])
        pred_cls = torch.sigmoid(x[:, :, :, :, 5:5+classes])

        # prediction values that are going to be used for the original image
        # we need to detach them from the graph as we don't need to backproparate
        # on them
        predictions = x.clone().detach()
        # broadcasting (B, A, G, G) + (1, 1, G, 1)
        # broadcasting (B, A, G, G) + (1, 1, 1, G)
        # For now, we are not going to multiply them by stride since
        # we need them in make_targets
        predictions[:, :, :, :, 0] = pred_x + c_x
        predictions[:, :, :, :, 1] = pred_y + c_y
        # broadcasting (1, A, 1, 1, 2) * (B, A, G, G, 2)
        predictions[:, :, :, :, 2:4] = p_wh * torch.exp(pred_wh)
        predictions[:, :, :, :, 4] = pred_obj
        predictions[:, :, :, :, 5:5+classes] = pred_cls

        if targets is not None:
            # We prepare targets at each scale as it depends on the
            # number of grid cells. So, we cannot do this once in,
            # let's say, __init__().
            ious, cls_mask, obj_mask, noobj_mask, gt_x, gt_y, gt_w, gt_h, gt_cls, gt_obj = self.make_targets(
                predictions, targets, anchors_tensor, self.ignore_thresh, device
            )
            # Loss
            # (todo: replace it with a separate function)
            # (1) Localization loss
            loss_x = self.mse_loss(pred_x[obj_mask], gt_x[obj_mask])
            loss_y = self.mse_loss(pred_y[obj_mask], gt_y[obj_mask])
            loss_w = self.mse_loss(pred_wh[..., 0][obj_mask], gt_w[obj_mask])
            loss_h = self.mse_loss(pred_wh[..., 1][obj_mask], gt_h[obj_mask])
            # (2) Confidence loss
            loss_conf_obj = self.bce_loss(pred_obj[obj_mask], gt_obj[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_obj[noobj_mask], gt_obj[noobj_mask])
            loss_conf = self.obj_coeff * loss_conf_obj + self.noobj_coeff * loss_conf_noobj
            # (3) Classification loss
            loss_cls = self.bce_loss(pred_cls[obj_mask], gt_cls[obj_mask])
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            self.metrics_dict = self.calculate_metrics_at_this_scale(
                pred_obj, ious, cls_mask, obj_mask, noobj_mask, gt_obj
            )
            self.metrics_dict['G'] = G
            self.metrics_dict['loss'] = loss.item()
            self.metrics_dict['loss_x'] = loss_x.item()
            self.metrics_dict['loss_y'] = loss_y.item()
            self.metrics_dict['loss_w'] = loss_w.item()
            self.metrics_dict['loss_h'] = loss_h.item()
            self.metrics_dict['loss_conf'] = loss_conf.item()
            self.metrics_dict['loss_cls'] = loss_cls.item()

        # multiplying by stride only now because the make_targets func
        # required predictions to be in grid values
        predictions[:, :, :, :, :4] = predictions[:, :, :, :, :4] * stride
        # for NMS: (B, A, G, G, 5+classes) -> (B, A*G*G, 5+classes)
        predictions = predictions.view(B, G*G*num_anchs, num_feats)

        return predictions, loss

    def make_targets(self, predictions, targets, anchors, ignore_thresh, device):
        '''
        Builds the neccessary g.t. masks and bbox attributes. It is expected to be
        used at each scale of Darknet.

        Arguments
        ---------
        predictions: torch.FloatTensor
            Predictions \in (B, A, Gs, Gs, 5+classes) where A - num of anchors
            Gs - number of grid cells at the respective scale.
            Note: predictions have to be inputed before multiplying by stride
                  (= input_img_size // number_of_grid_cells_on_one_side).
        targets: torch.FloatTensor
            Ground Truth bboxes and their classes. The tensor has
            the number of rows according to the number of g.t. bboxes
            in the whole batch and 6 columns:
                - the image idx within the batch;
                - the g.t. label corresponding to this bbox;
                - center coordinates x, y \in (0, 1);
                - bbox width and height
            Note: the coordinates account for letter padding which means
                  that the coordinates for the center are shifted accordingly
        anchors: torch.FloatTensor
            A tensor with anchors of size (num_anchs, 2) with width and height
            lengths in the number of grid cells they overlap at the current
            scale. Anchors are calculated as the config anchors divided by the stride
            (= input_img_size // number_of_grid_cells_on_one_side).
        ignore_thresh: float
            A threshold is a hyper-parameter that is used only when noobjectness
            mask (noobj_mask) is generated (see the code for insights).
        device: torch.device
            A device to put the tensors on.

        Outputs
        -------
        iou_scores: torch.FloatTensor
            A tensor of size (B, A, Gs, Gs), where Gs represents number of grid cells
            at the respective scale.
            Contains all zeros except for the IoUs at [img_idx, best_anchors, gj, gi]
            between:
                a) predicted bboxes (at the positions of anchors with the highest
                IoU with g.t. bboxes at a (gj, gi) grid-cell) and
                b) the g.t. (target) bboxes.
            In other words, we take a g.t. bbox, look at the anchor which fits it best,
            find the same exact location (gj, gi) and anchor (best_anchors)
            at image (img_idx) in the predictions and calculate IoU between them.
            At other predictions, which g.t. doesn't cover, it is 0.
            Used later for the estimation of metrics.
        class_mask: torch.FloatTensor
            A tensor of size (B, A, Gs, Gs), where Gs represents number of grid
            cells at the respective scale.
            Contains all zeros except for ones at [img_idx, best_anchors, gj, gi] if
            a predicted class at [img_idx, best_anchors, gj, gi] matches the g.t.
            label.
            Used later for the estimation of metrics.
        obj_mask: torch.ByteTensor
            A tensor of size (B, A, Gs, Gs), where Gs represents number of grid
            cells at the respective scale.
            Contains all zeros except for ones at [img_idx, best_anchors, gj, gi].
            The same as class_mask but less strict: it is one regardless of whether
            the predicted label matches the g.t. label.
            Used later for the estimation of the loss and metrics.
        noobj_mask: torch.ByteTensor
            A tensor of size (B, A, Gs, Gs), where Gs represents number of grid
            cells at the respective scale.
            A mask which is an opposite to obj_mask. It has zeros where obj_mask has
            ones and also where IoU betwee g.t. and anchors are higher than
            ignore_thresh.
            Used later for the estimation of the loss and metrics.
        gt_x, gt_y: torch.FloatTensor, torch.FloatTensor
            Tensors of size (B, A, Gs, Gs), where Gs represents number of grid
            cells at the respective scale.
            Contain the values in [0, 1] at [img_idx, best_anchors, gj, gi] at x and y
            which represent the position of the center of the g.t. bbox w.r.t
            the top-left corner of the corresponding cell (gi, gi).
            For example, if the g.t. bbox' center coordinates are (3.57, 12.3), the
            tx and ty at [img_idx, best_anchors, gj, gi] are going to be (0.57 and 0.3).
            Used later for the estimation of the loss.
        gt_w, gt_h: torch.FloatTensor, torch.FloatTensor
            Tensors of size (B, A, Gs, Gs), where Gs represents number of grid
            cells at the respective scale.
            Contain the values in (log(0), log(Gs)] respectively
            at [img_idx, best_anchors, gj, gi] for both width and height respectively
            others are 0s and represent the log-transformation of the g.t. coefficient
            that is used to multiply the anchors to fit the dimensions of the g.t. bboxes.
            Used later for the estimation of the loss.
        gt_cls: torch.FloatTensor
            A tensor of size !(B, num_anchs, G, G, num_classes)!.
            One-hot-encoding of the g.t. label at [img_idx, best_anchors, gj, gi].
            Used later for the estimation of the loss.
        gt_obj: torch.FloatTensor
            A tensor of size (B, A, Gs, Gs), where Gs represents number of grid
            cells at the respective scale.
            Contains the same info as in obj_mask but it is of a
            different type. Used to make the code more readable when loss is
            calculated.

            # TODO: gt_cls.sum(dim=-1) == obj_mask??
        '''
        B, num_anchs, G, G, num_feats = predictions.size()
        classes = num_feats - 5

        # create the placeholders
        noobj_mask = torch.ones(B, num_anchs, G, G, device=device).byte()
        obj_mask = torch.zeros_like(noobj_mask).byte()
        class_mask = torch.zeros_like(noobj_mask).float()
        iou_scores = torch.zeros_like(noobj_mask).float()
        gt_x = torch.zeros_like(noobj_mask).float()
        gt_y = torch.zeros_like(noobj_mask).float()
        gt_w = torch.zeros_like(noobj_mask).float()
        gt_h = torch.zeros_like(noobj_mask).float()
        gt_cls = torch.zeros(B, num_anchs, G, G, classes, device=device).float()

        # image index within the batch, the g.t. label of an object on the image
        img_idx, gt_class_int = targets[:, :2].long().t()
        # ground truth center coordinates and bbox dimensions
        # since the target bbox coordinates are in (0, 1) but we predict on one
        # of the 3 scales (YOLOv3) we multiply it by the number of grid cells
        # So, the cxy will represent the position of the center on a grid.
        # Similarly, the size sizes are also scalled to grid size
        # todo: rename these variables but gt is already used
        cxy = targets[:, 2:4] * G
        bwh = targets[:, 4:] * G
        # ious between scaled anchors (anchors_from_cfg / stride) and gt bboxes
        gt_anchor_ious = iou_vectorized(anchors, bwh, without_center_coords=True)
        # selecting the best anchors for the g.t. bboxes
        best_ious, best_anchors = gt_anchor_ious.max(dim=0)

        cx, cy = cxy.t()
        bw, bh = bwh.t()
        # remove a decimal part -> gi, gj point to the top left coord
        # for a grid cell to which an object will correspond
        gi, gj = cxy.long().t()
        # helps with RuntimeError: CUDA error: device-side assert triggered
        # This aims to avoid gi[i] and gj[i] exceeding bound of size[2, 3]
        # of noobj_mask.
        gi[gi < 0] = 0
        gj[gj < 0] = 0
        gi[gi > G - 1] = G - 1
        gj[gj > G - 1] = G - 1
        # update the obj and noobj masks.
        # the noobj mask has 0 where obj mask has 1 and where IoU between
        # g.t. bbox and anchor is higher than ignore_thresh
        obj_mask[img_idx, best_anchors, gj, gi] = 1
        noobj_mask[img_idx, best_anchors, gj, gi] = 0

        for i, gt_anchor_iou in enumerate(gt_anchor_ious.t()):
            noobj_mask[img_idx[i], gt_anchor_iou > ignore_thresh, gj[i], gi[i]] = 0

        # coordinates of a center with respect to the top-left corner of a grid cell
        gt_x[img_idx, best_anchors, gj, gi] = cx - cx.floor()
        gt_y[img_idx, best_anchors, gj, gi] = cy - cy.floor()
        # since yolo predicts the coefficients (log of coefs actually, see exp(tw)
        # in the paper) that will be used to multiply with anchor sides,
        # for ground truth side lengths, in turn, we should apply log transformation.
        # In other words, for the loss we need to compare the values of the same scale.
        # Suppose, yolo predicts coefficient that is goint to be used to scale the anchors
        # in log-scale first, then we apply exponent which makes them to be in regular
        # scale (see the yolo layer in darknet.py). Since, we are going to need only
        # the log-scale values before it is transformed to regular scale for the loss
        # calculation, we also, then, need to transform the g.t. coefficient to log-scale
        # from the regular scale, hence, the log here.
        gt_w[img_idx, best_anchors, gj, gi] = torch.log(bw / anchors[best_anchors][:, 0] + self.EPS)
        gt_h[img_idx, best_anchors, gj, gi] = torch.log(bh / anchors[best_anchors][:, 1] + self.EPS)
        # one-hot encoding of a label
        gt_cls[img_idx, best_anchors, gj, gi, gt_class_int] = 1
        # compute label correctness and iou at best anchor -- we will use them to
        # calculate the metrics outside during the training loop.
        # Extracting the labels from the prediciton tensor
        pred_xy_wh = predictions[img_idx, best_anchors, gj, gi, :4]
        pred_class_probs = predictions[img_idx, best_anchors, gj, gi, 5:5+classes]
        _, pred_class_int = torch.max(pred_class_probs, dim=-1)
        class_mask[img_idx, best_anchors, gj, gi] = (pred_class_int == gt_class_int).float()
        # since iou_vectorized returns IoU between all possible pairs but we
        # need to know only IoU between corresponding rows (pairs) we select only diagonal
        # elements
        iou_scores[img_idx, best_anchors, gj, gi] = iou_vectorized(
            pred_xy_wh,
            targets[:, 2:6] * G
        ).diag()
        # ground truth objectness
        gt_obj = obj_mask.float()

        return iou_scores, class_mask, obj_mask, noobj_mask, gt_x, gt_y, gt_w, gt_h, gt_cls, gt_obj

    def calculate_metrics_at_this_scale(self, pred_obj, iou_scores, class_mask, obj_mask, noobj_mask, gt_obj):
        '''
        TODO ONCE TESTED
        '''
        # obj_mask has 1s only where there is a g.t. in that cell
        # cls_mask is the same as obj_mask but more "strics": it
        # has 1s where the is a g.t. AND the predicted label
        # matches the g.t.
        # Therefore, it measures how well the model predicts labels
        # at positions with g.t.
        # Note: we cannot simplify it as this: cls_mask.mean()
        # since it would underestimate the accuracy
        accuracy = 100 * class_mask[obj_mask].mean()
        # measures how confident the model is in its predictions
        # at positions with g.t. objects and elsewhere
        conf_obj = pred_obj[obj_mask].mean()
        conf_noobj = pred_obj[noobj_mask].mean()
        # create a mask with 1s where objectness score
        # is high enough, 0s elsewhere
        pred_obj50_mask = (pred_obj > 0.5).float()
        # create a mask with 1s where IoU between the best fitting
        # anchors for g.t. and predictions at the positions with g.t.
        iou50_mask = (iou_scores > 0.5).float()
        iou75_mask = (iou_scores > 0.75).float()
        # has 1s where the predicted label matches
        # the g.t. one and if the predicted objectness score is
        # high enough (0.5)
        detected_mask = pred_obj50_mask * class_mask * gt_obj
        # pred_obj50_mask.sum() = number of confident predictions
        all_detections = pred_obj50_mask.sum()
        # obj_mask.sum() = number of g.t. objects
        all_ground_truths = obj_mask.sum()

        # precision = TP / (TP + FP) = TP / all_detections
        precision = (iou50_mask * detected_mask).sum() / (all_detections + self.EPS)
        # recall = TP / (TP + FN) = TP / all_ground_truths
        recall50 = (iou50_mask * detected_mask).sum() / (all_ground_truths + self.EPS)
        recall75 = (iou75_mask * detected_mask).sum() / (all_ground_truths + self.EPS)

        metrics_dict = {
            'accuracy': accuracy.item(),
            'conf_obj': conf_obj.item(),
            'conf_noobj': conf_noobj.item(),
            'precision': precision.item(),
            'recall50': recall50.item(),
            'recall75': recall75.item(),
        }

        return metrics_dict


class Darknet(nn.Module):
    '''Darknet model (YOLO v3)'''

    def __init__(self, cfg_path):
        '''
        Argument
        --------
        cfg_path: str
            A path to config file.
            Example: github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        '''
        super(Darknet, self).__init__()
        self.layers_info = parse_cfg(cfg_path)
        self.net_info, self.layers_list = self.create_layers(self.layers_info)
        # take the number of classes from the last (yolo) layer of the network.
        self.classes = self.layers_list[-1][0].classes
        self.model_width = self.layers_list[-1][0].model_width
        # shortcut is using output[i-1] instead of x check whether works with x
        # changing predictions in the NMS loop make sure that it is not used later
        # not adding +1 in nms
        # loss: w and h aren`t put through sqroot

    def forward(self, x, targets=None, device=torch.device('cpu')):
        '''
        Arguments
        ---------
        x: torch.FloatTensor
            An image of size (B, C, H, W).
        targets: torch.FloatTensor
            Ground Truth bboxes and their classes. The tensor has
            the number of rows according to the number of g.t. bboxes
            in the whole batch and 6 columns:
                - the image idx within the batch;
                - the g.t. label corresponding to this bbox;
                - center coordinates x, y \in (0, 1);
                - bbox width and height
            Note: the coordinates account for letter padding which means
                  that the coordinates for the center are shifted accordingly
        device: torch.device
            The device to use for calculation: torch.device('cpu'), torch.device('cuda:?')

        Output
        ------
        x: torch.FloatTensor
            A tensor of size (B, P, 5+classes) with predictions.
            B -- batch size; P -- number of predictions for an image,
            i.e. 3 scales and 3 anchor boxes and
            For example: P = (13*13 + 26*26 + 52*52) * 3 = 10647;
            5 + classes -- (cx, cy, w, h, obj_score, {prob_class}).

        total_loss: torch.FloatTensor
            If targets is specified.
            Accumulated loss for each scale.
        '''
        # since we are use resizing augmentation: model_width != input_width
        input_width = x.size(-1)
        # cache the outputs from every layer. Used in route and shortcut layers
        outputs = []
        # the prediction of Darknet is the stacked predictions from all scales
        predictions = []
        # initialize the loss that is going to be added to the
        # total loss at each scale
        loss = 0

        for i, layer in enumerate(self.layers_list):
            # i+1 because 0th is net_info
            name = self.layers_info[i+1]['name']

            if name in ['convolutional', 'upsample']:
                x = layer(x)

            elif name == 'shortcut':
                # index which is used for shortcut (usually '-3')
                x = outputs[-1] + outputs[layer[0].frm]
                # x = x + outputs[layer.frm]

            elif name == 'route':
                to_cat = [outputs[route_idx] for route_idx in layer[0].routes]
                x = torch.cat(to_cat, dim=1)

            elif name == 'yolo':
                x, loss = layer[0](x, targets, device, input_width)
                # increment loss. Loss is calculated at each YOLOLayer
                loss += loss
                predictions.append(x)

            # after each layer we append the current output to the outputs list
            outputs.append(x)

        # the list of predictions is now concatenated in one tensor
        predictions = torch.cat(predictions, dim=1)

        if targets is None:
            assert loss == 0, f'loss = 0 if targets is None, loss: {loss}'

        return predictions, loss

    def save_model(self, log_path, epoch, optimizer):
        '''
        Saves the model to specified path. If doesn't exist creates the folder.

        Arguments
        ---------
        log_path: str
            Path to save the model. It is expected to be the Tensorboard path
        epoch: int
            An epoch number at which the model is saved
        optimizer: torch.optim??
            The state of an optimizer
        '''
        dict_to_save = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        # in case TBoard was not been defined, make a logdir
        os.makedirs(log_path, exist_ok=True)
        path_to_save = os.path.join(log_path, 'bestmodel.pt')
        torch.save(dict_to_save, path_to_save)

    def load_weights(self, weight_file):
        '''
        Loads weights from the original weight file.

        Argument
        --------
        weight_file: str
            A part to weights file
        '''
        r_file = open(weight_file, 'rb')
        # header consists on version numbers (major, minor, subversion)
        # and number of images seen by model during training
        header = np.fromfile(r_file, dtype=np.int32, count=5)
        # the rest are weights in a form of a vector (not matrix)
        weights = np.fromfile(r_file, dtype=np.float32)

        idx = 0

        for i, layer in enumerate(self.layers_list):
            # i+1 because 0th is net_info
            name = self.layers_info[i+1]['name']

            if name == 'convolutional':

                # some conv layers doesn't have bn layers
                try:
                    self.layers_info[i+1]['batch_normalize']
                    conv, bn = self.layers_list[i][:2]

                    # 1. load and convert the selected weights to a Tensor
                    # 2. reshape loaded weights as the current ones
                    # 3. replace the current weights with the loaded ones
                    # 4. increment the current counter of read weights

                    # num of bn weights
                    bn_weigth_num = bn.bias.numel()

                    # bn biases
                    bn_b = torch.Tensor(weights[idx:idx+bn_weigth_num])
                    bn_b = bn_b.view_as(bn.bias.data)
                    bn.bias.data.copy_(bn_b)
                    idx += bn_weigth_num

                    # bn weights
                    bn_w = torch.Tensor(weights[idx:idx+bn_weigth_num])
                    bn_w = bn_w.view_as(bn.weight.data)
                    bn.weight.data.copy_(bn_w)
                    idx += bn_weigth_num

                    # bn running mean
                    bn_running_mean = torch.Tensor(weights[idx:idx+bn_weigth_num])
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean.data)
                    bn.running_mean.data.copy_(bn_running_mean)
                    idx += bn_weigth_num

                    # bn running var
                    bn_running_var = torch.Tensor(weights[idx:idx+bn_weigth_num])
                    bn_running_var = bn_running_var.view_as(bn.running_var.data)
                    bn.running_var.data.copy_(bn_running_var)
                    idx += bn_weigth_num

                    # conv weights (no need to load biases if bn is used)
                    conv_w_num = conv.weight.numel()
                    conv_w = torch.Tensor(weights[idx:idx+conv_w_num])
                    conv_w = conv_w.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_w)
                    idx += conv_w_num

                except KeyError:
                    conv = self.layers_list[i][0]

                    # conv biases
                    conv_b_num = conv.bias.numel()
                    conv_b = torch.Tensor(weights[idx:idx+conv_b_num])
                    conv_b = conv_b.view_as(conv.bias.data)
                    idx += conv_b_num

                    # conv weights
                    conv_w_num = conv.weight.numel()
                    conv_w = torch.Tensor(weights[idx:idx+conv_w_num])
                    conv_w = conv_w.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_w)
                    idx += conv_w_num

                    # print('no bn detected at {}'.format(i))

    def create_layers(self, layers_info):
        '''An auxiliary fuction that creates a ModuleList given layers_info

        Argument
        --------
        layers_info: list:
            a list containing net info (0th element) and info for each
            layer (module) specified in the config (1: elements).

        Outputs
        ------
        net_info, layers_list: (dict, torch.nn.ModuleList)
            net_info contains model's info.
            layers_list contains list of layers.
        '''
        # the first element is not a layer but the network info (lr, batchsize,  ...)
        net_info = layers_info[0]
        # init. the modulelist instead of a list to add all parameters to nn.Module
        layers_list = nn.ModuleList()
        # cache the # of filters as we will need them in Conv2d
        # it starts with the number of channels specified in net info, often = to 3 (RGB)
        filters_cache = [int(net_info['channels'])]

        # print('WARNING: sudivisions of a batch aren`t used in contrast to the original cfg')

        for i, layer_info in enumerate(layers_info[1:]):
            # we initialize sequential as a layer may have conv, bn, and activation
            layer = nn.Sequential()
            name = layer_info['name'] # conv, upsample, route, shortcut, yolo

            if name == 'convolutional':

                # extract arguments for the layer
                in_filters = filters_cache[-1]
                out_filters = int(layer_info['filters'])
                kernel_size = int(layer_info['size'])
                pad = (kernel_size - 1) // 2 if int(layer_info['pad']) else 0
                stride = int(layer_info['stride'])

                # some layers doesn't have BN
                try:
                    layer_info['batch_normalize']
                    conv = nn.Conv2d(in_filters, out_filters, kernel_size, stride, pad, bias=False)
                    layer.add_module('conv_{}'.format(i), conv)
                    layer.add_module('bn_{}'.format(i), nn.BatchNorm2d(out_filters))

                except KeyError:
                    conv = nn.Conv2d(in_filters, out_filters, kernel_size, stride, pad)
                    layer.add_module('conv_{}'.format(i), conv)

                # activation. if 'linear': no activation
                if layer_info['activation'] == 'leaky':
                    layer.add_module('leaky_{}'.format(i), nn.LeakyReLU(0.1))

            elif name == 'upsample':
                # extract arguments for the layer
                stride = int(layer_info['stride'])
                layer.add_module(
                    'upsample_{}'.format(i),
                    nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
                )

            # here we need to deal only with the number of filters
            elif name == 'route':
                # route can have one, two, or more sources
                # first, let's make them to be ints
                routes = [int(route) for route in layer_info['layers'].split(',')]
                # then, sum the number of filters from at each mentioned layer
                out_filters = sum([filters_cache[route] for route in routes])
                # # add the dummy layer to the list
                # layer.add_module('route_' + i, EmptyLayer())
                # add the route layer to the modulelist
                layer.add_module('route_{}'.format(i), RouteLayer(routes))

            # in forward() we will need to add the output of a previous layer, nothing to do here
            elif name == 'shortcut':
                # from which layer to use the shortcut
                frm = int(layer_info['from'])
                # add the shortcut layer to the modulelist
                layer.add_module('shortcut_{}'.format(i), ShortcutLayer(frm))

            # yolo layer
            elif name == 'yolo':
                # extract arguments for the layer
                classes = int(layer_info['classes'])
                num = int(layer_info['num'])
                jitter = float(layer_info['jitter'])
                ignore_thresh = float(layer_info['ignore_thresh'])
                truth_thresh = float(layer_info['truth_thresh'])
                random = int(layer_info['random']) # int??
                in_width = int(net_info['width'])

                # masks tells the dector which anchor to use (form: '6,7,8')
                masks = [int(mask) for mask in layer_info['mask'].split(',')]
                # select anchors (form: 10,13,16,30,33,23,30,61,62,45 -- 5 pairs)
                # first extract the coordinates
                coords = [int(coord) for coord in layer_info['anchors'].split(',')]
                # make anchors (tuples)
                anchors = list(zip(coords[::2], coords[1::2]))
                # select anchors that belong to mask
                anchors = [anchors[mask] for mask in masks]

                # add the yolo layer to the list
                yolo = YOLOLayer(anchors, classes, num, jitter, ignore_thresh,
                                 truth_thresh, random, in_width)
                layer.add_module('yolo_{}'.format(i), yolo)


            # append the layer to the modulelist
            layers_list.append(layer)
            # append number of filter to filter_cache at each iteration (inc. yolo and shorcut)
            filters_cache.append(out_filters)

        # print('INFO: `make_layers` returns `net_info` as well. Check whether it is necessary')
        return net_info, layers_list
