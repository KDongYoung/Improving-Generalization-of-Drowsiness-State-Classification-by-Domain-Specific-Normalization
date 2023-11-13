'''ResNet-1D in PyTorch.
Dong-Kyun Han 2020/09/17
dkhan@korea.ac.kr

Reference:
[1] K. He, X. Zhang, S. Ren, J. Sun
    "Deep Residual Learning for Image Recognition," arXiv:1512.03385
[2] J. Y. Cheng, H. Goh, K. Dogrusoz, O. Tuzel, and E. Azemi,
    "Subject-aware contrastive learning for biosignals,"
    arXiv preprint arXiv :2007.04871, Jun. 2020
[3] D.-K. Han, J.-H. Jeong
    "Domain Generalization for Session-Independent Brain-Computer Interface,"
    in Int. Winter Conf. Brain Computer Interface (BCI),
    Jeongseon, Republic of Korea, 2020.
'''

import torch
import torch.nn as nn

from utils.util import Conv1d, TwoInputSequential

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, f_res=False, stride=1, 
                downsample=None, norm_layer=None, num_domain=None, track_running=True, batch_size=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        if num_domain==0: # norm_layer=nn.BatchNorm1d
            self.bn0 = norm_layer(planes, track_running_stats=track_running)
            self.bn1 = norm_layer(planes, track_running_stats=track_running)
        else:
            self.bn0 = norm_layer(planes, num_domain=num_domain, track_running_stats=track_running, dim=1, batch_size=batch_size)
            self.bn1 = norm_layer(planes, num_domain=num_domain, track_running_stats=track_running, dim=1, batch_size=batch_size)
            
        self.elu = nn.GELU()
        self.dropdout0 = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.dropdout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

        self.downsample = downsample
        self.stride = stride
        
        self.f_res=f_res
        
    def forward(self, x, domain_label=None):
        
        if domain_label is None: # original BN
            
            identity = x.clone() # out0
            
            out = self.bn1(self.conv1(x))
            out = self.dropdout1(self.elu(out))

            out = self.bn0(self.conv2(out))
            out = self.dropdout0(self.elu(out))

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            
            return out

        else: 
            
            identity = x.clone() # out0
            
            out, _ = self.bn1(self.conv1(x), domain_label)
            out = self.dropdout1(self.elu(out))

            out, _ = self.bn0(self.conv2(out), domain_label)
            out = self.dropdout0(self.elu(out))
            
            if self.downsample is not None:
                identity, _ = self.downsample(x, domain_label)
            out += identity

            return out, domain_label

class reResnet(nn.Module):
    def __init__(self, args, layers, kernel_sizes, planes, strides, num_domain, batch_norm=True, batch_norm_alpha=0.1):
        super(reResnet, self).__init__()

        self.batch_size=args['batch_size']
        self.track_running= args['track_running']
        self.layer_len=len(layers)
        self.f_res=args["f_res"]
        self.norm_type=args["norm_type"]
        
        num_classes = args['n_classes']
        input_ch=args['n_channels']
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_ch4 = 200
        self.num_hidden = 1024

        self.dilation = 1
        self.groups = 1
        self.base_width = input_ch
        norm_layer = nn.BatchNorm1d
        self.num_bn=0
        
        if self.norm_type=="multibn":
            from utils.BN.MultiBN import MultiBN
            self._norm_layer = MultiBN
            self.num_bn=num_domain
                    
        elif self.norm_type=="bn":
            self._norm_layer = nn.BatchNorm1d
                
        if self.norm_type=="bn" or self.norm_type=="in" or self.norm_type=="ibn":
            self.Sequential=nn.Sequential
            self.Conv=nn.Conv1d
        else:
            self.Sequential=TwoInputSequential
            self.Conv=Conv1d
            
        self.inplanes = 32
        self.conv1 = nn.Conv1d(input_ch, self.inplanes, kernel_size=13, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.elu = nn.GELU()

        block = BasicBlock

        layer=[]
        for i in range(len(layers)):
            layer.append(self._make_layer(block, planes[i], kernel_sizes[i], layers[i], stride=strides[i], layer_num=i))

        self.layers = nn.ModuleList(layer)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, kernel_size, blocks, stride=1, layer_num=0, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.norm_type=="bn" or self.norm_type=="in" or self.norm_type=="ibn":
                downsample = self.Sequential(
                    self.Conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion, track_running_stats=self.track_running))
            else:
                downsample = self.Sequential(
                    self.Conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion, num_domain=self.num_bn, track_running_stats=self.track_running, dim=1, batch_size=self.batch_size))

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, self.f_res, stride, downsample, 
                            norm_layer, self.num_bn, self.track_running, self.batch_size))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, self.f_res, norm_layer=norm_layer, 
                                num_domain=self.num_bn, track_running=self.track_running, batch_size=self.batch_size))
            
        return self.Sequential(*layers)

    def forward(self, x, target_y=None, domain=None):
        
        if domain==None: domain=torch.zeros(x.shape[0]).to(torch.int32)
        
        if self.norm_type=="bn" or self.norm_type=="in" or self.norm_type=="ibn":
            output, target_y = self.forward_bn(x, target_y)
        else:
            output, target_y = self.forward_multibn(x, target_y, domain)
        return output, target_y
    
    def forward_bn(self, x, target_y=None):
        x = x.squeeze(1)        
        x = self.conv1(x)
        
        for i in range(self.layer_len):
            
            if i<self.layer_len-2:
                x = self.layers[i](x)       
            else:
                x = self.maxpool(x)
                x = self.layers[i](x)   
                
        x = self.elu(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, target_y
    
    def forward_multibn(self, x, target_y=None, domain=None):
        x = x.squeeze(1)
        x = self.conv1(x)
        
        for i in range(self.layer_len):
            
            if i<self.layer_len-2:
                x, _ = self.layers[i](x, domain)   
            else:
                x = self.maxpool(x)
                x, _ = self.layers[i](x, domain)  

        x = self.elu(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, target_y
        

def resnet_18(args, num_domain):
    layers = [2,2,2,2]
    kernel_sizes = [3, 3, 3, 3]
    planes = [32, 64, 128, 256]
    strides = [1, 1, 2, 2]
    return reResnet(args, layers, kernel_sizes, planes, strides, num_domain)

def resnet_8(args, num_domain):
    layers = [1,1,1]
    kernel_sizes = [11, 9, 7]
    planes = [32, 128, 256]
    strides = [1, 1, 2]
    return reResnet(args, layers, kernel_sizes, planes, strides, num_domain)