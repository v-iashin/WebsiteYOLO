import torch
from torch import nn

from utils import parse_cfg

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
    the index of the output to use the shortcut from'''
    
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
        super(Darknet, self).__init__()
        self.layers_info = parse_cfg(cfg_path)
        self.net_info, self.layers_list = self.create_layers(self.layers_info)
        print('shortcut is using output[i-1] instead of x check whether works with x')
        print('NOTE THAT CONV BEFORE YOLO USES (num_classes filters) * num_anch')
        
    def forward(self, x, device):
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
                a, b = torch.meshgrid(grid, grid)
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

                # transform the predictions
                x[:, :, 0:2] = torch.sigmoid(x[:, :, 0:2]) + cxy
                x[:, :, 2:4] = pwh * torch.exp(x[:, :, 2:4])
                x[:, :, 4] = torch.sigmoid(x[:, :, 4]) * stride
                x[:, :, 5:5+classes] = torch.sigmoid((x[:, :, 5:5+classes]))

                # add new predictions to the list of predictions from all scales
                # if variable does exist
                try:
                    predictions = torch.cat((predictions, x), dim=1)

                except NameError:
                    predictions = x

            outputs.append(x)

        return predictions
    
    def create_layers(self, layers_info):
        '''An auxiliary fuction that creates a ModuleList given layers_info
        
        Input: layers_info (list): a list containing net info (0th element) and 
                                   info for each layer (module) specified in the config
                                   (1: elements).
        Outputs: net_info, layers_list (tuple): net_info (dict) contains model info
                                                layers_list (nn.ModuleList) contains list of
                                                layers.'''
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

                # make conv module and add it to the sequential
                conv = nn.Conv2d(in_filters, out_filters, kernel_size, stride, pad)
                layer.add_module('conv_{}'.format(i), conv)

                # some layers doesn't have BN
                # TODO: fix the bn if possible (bias)
                try:
                    layer_info['batch_normalize']
                    layer.add_module('bn_{}'.format(i), nn.BatchNorm2d(out_filters))

                except KeyError:
                    print('to del, this message should be printed 3ish times')
                    pass

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
