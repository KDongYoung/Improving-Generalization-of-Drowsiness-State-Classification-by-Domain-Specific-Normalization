import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single

from collections import OrderedDict
import operator
from itertools import islice


def to_one_hot(inp,num_classes):
    '''one hot encoding'''
    y_onehot = torch.FloatTensor(inp.size(0), num_classes) 
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return y_onehot


###########################
# For model construction
###########################
class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            False, _single(0), groups, bias, "zeros")

    def forward(self, input, domain_label):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            False, _single(0), groups, bias, "zeros")

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label


class TwoInputSequential(nn.Module):
    """A sequential container forward with two inputs."""

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            if str(module).split("(")[0] == "Conv2d" or str(module).split("(")[0] == "Conv1d" or 'Multi' in str(module).split("(")[0]: # == "MultiBatchNorm":
                input1, input2 = module(input1, input2)
                self.domain_label=input2
            elif str(module).split("(")[0] == "BasicBlock":
                input1, input2 = module(input1, input2)
                self.domain_label=input2
            elif "BN" in str(module).split("(")[0]:
                input1, input2 = module(input1, input2)
            else:
                input1 = module(input1)
                input2 = self.domain_label
            
        return input1, input2