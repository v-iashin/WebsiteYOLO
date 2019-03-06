import numpy as np

import torch
from torch import nn

from utils import parse_cfg, iou_vectorized

# class EmptyLayer(nn.Module):
#     '''A dummy layer for "route" and "shortcut" layers'''
    
#     def __init__(self):
#         super(EmptyLayer, self).__init__()
        
class RouteLayer(nn.Module):
    '''Route layer only saves the indices to access the outputs from the previous layers'''
    
    def __init__(self, routes):
        super(RouteLayer, self).__init__()
        self.routes = routes
        
class ShortcutLayer(nn.Module):
    '''Similarly to Routelayer shortcut layer functions as a dummy layer and only saves
    the index of the layer to use the shortcut from'''
    
    def __init__(self, frm):
        super(ShortcutLayer, self).__init__()
        self.frm = frm
        
class YOLOLayer(nn.Module):
    '''Similarly to the previous layers, YOLO layer in defined'''
    
    def __init__(self, anchors, classes, num, jitter, ignore_thresh, truth_thresh, random, in_width):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.num = num
        self.jitter = jitter
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.random = random
        self.in_width = in_width
        
        
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
        print('shortcut is using output[i-1] instead of x check whether works with x')
        print('NOTE THAT CONV BEFORE YOLO USES (num_classes filters) * num_anch')
        print('changing predictions in the nms loop make sure that it is not used later')
        print('not adding +1 in nms')
        
    def forward(self, x, device):
        '''
        Arguments
        ---------
        x: torch.FloatTensor
            An image of size (B, C, H, W).
        device: torch.device
            The device to use for calculation: torch.device('cpu'), torch.device('gpu:0')
            
        Output
        ------
        x: torch.FloatTensor
            A tensor of size (B, P, 5+classes) with predictions after filtering using
            objectness score
            B -- batch size; P -- number of predictions for an image, 
            i.e. 3 scales and 3 anchor boxes and
            For example: P = (13*13 + 26*26 + 52*52) * 3 = 10647;
            5 + classes -- (cx, cy, w, h, obj_score, {prob_class}).
        '''
        # cache the outputs for route and shortcut layers
        outputs = []

        for i, layer in enumerate(self.layers_list):
            # i+1 because 0th is net_info
            name = self.layers_info[i+1]['name']

            if name in ['convolutional', 'upsample']:
                x = layer(x)

            elif name == 'shortcut':
                # index which is used for shortcut (usually '-3')
                x = outputs[-1] + outputs[layer[0].frm]
        #         x = x + outputs[layer.frm]

            elif name == 'route':
                to_cat = [outputs[route_idx] for route_idx in layer[0].routes]
                x = torch.cat(to_cat, dim=1)

            elif name == 'yolo':
                # input size: (B, (4+1+classes)*num_achors=255, Gi, Gi)
                B, C, w, h = x.size()
                # read layer's info
                anchors_list = layer[0].anchors
                classes = layer[0].classes
                in_width = layer[0].in_width
                num_anchs = len(anchors_list)
                # bbox coords + obj score + class scores
                num_feats = 4 + 1 + classes

                # transform the predictions
                # (B, ((4+1+classes)*num_achors), Gi, Gi)
                # -> (B, Gi*Gi*num_anchors, (4+1+classes))
                x = x.view(B, num_anchs, num_feats, w, h)
                x = x.permute(0, 3, 4, 1, 2).contiguous() # (B, w, h, num_anchs, num_feats)
                x = x.view(B, h*w*num_anchs, num_feats)

                # To calc predictions, first we need to add a center offset (cx, cy).
                # To achieve this we need to create two columns with every possible 
                # combination of two numbers from 0 to num_grid and
                # then add it to the center offset predictions at 0 and 1 indices
                # in x.
                grid = torch.arange(w)
                # numpy and torch behaves differently in meshgrid -> b, a
                b, a = torch.meshgrid(grid, grid)
                cx = a.type(torch.FloatTensor).view(-1, 1) # 0, 0, ..., 12, 12
                cy = b.type(torch.FloatTensor).view(-1, 1) # 0, 1, ..., 11, 12
                # cxy.shape is (1, *, 2) where (:, :, 0) is an offset for cx, (:, :, 1) -- for cy
                cxy = torch.cat((cx, cy), dim=1).repeat(1, num_anchs).view(-1, 2).unsqueeze(0)
                cxy = cxy.to(device)

                # to calc the offsets for bbox size we need to scale anchors
                stride = in_width // w
                anchors_list = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors_list]
                anchors_tens = torch.FloatTensor(anchors_list)
                # pwh.shape is the same as cxy
                pwh = anchors_tens.repeat(w * w, 1).unsqueeze(0)
                pwh = pwh.to(device)

                # transform the predictions (center, size, objectness, class scores)
                x[:, :, 0:2] = (torch.sigmoid(x[:, :, 0:2]) + cxy) * stride
                x[:, :, 2:4] = (pwh * torch.exp(x[:, :, 2:4])) * stride
                x[:, :, 4] = torch.sigmoid(x[:, :, 4])
                x[:, :, 5:5+classes] = torch.sigmoid((x[:, :, 5:5+classes]))

                # add new predictions to the list of predictions from all scales
                # if variable does exist
                try:
                    predictions = torch.cat((predictions, x), dim=1)

                except NameError:
                    predictions = x

            outputs.append(x)

        return predictions
    
    def load_weights(self, weight_file):
        '''
        Loads weights from the weight file.
        
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

#                     print('no bn detected at {}'.format(i))

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

        print("WARNING: sudivisions of a batch aren't used in contrast to the original cfg" )
        print('we also can remove bias due to bn')

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
                layer.add_module('upsample_{}'.format(i), nn.Upsample(scale_factor=stride, mode='bilinear'))

            # here we need to deal only with the number of filters 
            elif name == 'route':
                # route can have one, two, or more sources
                # first, let's make them to be ints
                routes = [int(route) for route in layer_info['layers'].split(',')]
                # then, sum the number of filters from at each mentioned layer
                out_filters = sum([filters_cache[route] for route in routes])
        #         # add the dummy layer to the list
        #         layer.add_module('route_' + i, EmptyLayer())
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
                random = float(layer_info['random']) # float??
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

        print('make_layers returns net_info as well. check whether it"s necessary')
        return net_info, layers_list
    
    
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections


    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)