from torch import nn

class EmptyLayer(nn.Module):
    '''A dummy layer for "route" and "shortcut" layers'''
    
    def __init__(self):
        super(EmptyLayer, self).__init__()