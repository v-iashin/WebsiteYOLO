from torch import nn

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
        
class DetectionLayer(nn.Module):
    '''TODO the doc'''
    
    def __init__(self, anchors, classes, num, jitter, ignore_thresh, truth_thresh, random, in_width):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.num = num
        self.jitter = jitter
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.random = random
        self.in_width = in_width
        
        
class Darknet(nn.Module):
    '''TODO'''
    
    def __init__(self, layers_info):
        super(Darknet, self).__init__()
        self.layers_info = layers_info
    
    def create_layers(self):
        '''TODO'''
        # the first element is not a layer but the network info (lr, batchsize,  ...)
        net_info = self.layers_info[0]
        # init. the modulelist instead of a list to add all parameters to nn.Module
        layer_list = nn.ModuleList()

        print("WARNING: sudivisions of a batch aren't used in contrast to the original cfg" )

        for i, layer_info in enumerate(self.layers_info[1:]):
            # we initialize sequential as a layer may have conv, bn, and activation
            layer = nn.Sequential()
            # cache the # of filters as we will need them in Conv2d
            # it starts with the number of channels specified in net info, often = to 3 (RGB)
            filters_cache = [int(net_info['channels'])]
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
                try:
                    layer_info['batch_normalize']
                    layer.add_module('bn_{}'.format(i), nn.BatchNorm2d(out_filters))

                except KeyError:
                    print('to del, this message should be printed 3ish times')
                    pass

                # activation. if 'linear': no activation
                if layer_info['activation'] == 'leaky':
                    layer.add_module('leaky_{}'.format(i), nn.LeakyReLU(0.1))

                # add the number of filters to filters_cache
                filters_cache.append(out_filters)

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

                # add the number of filters to filters_cache
                filters_cache.append(out_filters)

            # in forward() we will need to add the output of a previous layer, nothing to do here
            elif name == 'shortcut':
                # from which layer to use the shortcut
                frm = int(layer_info['from'])
                # add the shortcut layer to the modulelist
                layer.add_module('shortcut_{}'.format(i), ShortcutLayer(frm))

            # detection layer
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
                coords = [int(coord) for coord in layer_info['anchors']]
                # make anchors (tuples)
                anchors = list(zip(coords[::2], coords[1::2]))
                # select anchors that belong to mask
                anchors = [anchors[mask] for mask in masks]

                # add the detector layer to the list
                detection = DetectionLayer(anchors, classes, num, jitter, ignore_thresh, 
                                           truth_thresh, random, in_width)
                layer.add_module('detector_' + i, detection)


            # append the layer to the modulelist
            layer_list.append(layer)

            print('make_layers returns net_info as well. check whether it"s necessary')
            return net_info, layer_list
