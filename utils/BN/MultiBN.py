import torch
import torch.nn as nn

'''
Domain-specific Batch Normalization in multi-domain setting

Reference:
    W.=G. Chang, T. You, S. Seo, S. Kwak, and B. Han
    Domain-specific batch normalization for unsupervised domain adaptation, CVPR 2019.

''' 
class MultiBN(nn.Module):

    def __init__(self, num_features, num_domain=10, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, dim=2, batch_size=16):
        super(MultiBN, self).__init__()

        self.num_domain=num_domain
        self.dim = dim
        self.batch_size=batch_size
        
        # Each Domain BN
        if dim==2:
            self.bns = nn.ModuleList(
                [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(self.num_domain)]) 
        elif dim==1:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) for _ in range(self.num_domain)]) 
            
    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if self.dim==1:
            if input.dim() != 3:
                raise ValueError('expected 3D input (got {}D input)'
                                .format(input.dim()))
        elif self.dim==2:
            if input.dim() != 4:
                raise ValueError('expected 4D input (got {}D input)'
                                .format(input.dim()))

    def forward(self, x, domain):
        self._check_input_dim(x)
        uniq_domain=domain.unique(return_counts=True)
        
        if not x.requires_grad: # eval state
            bs=1
        else:
            bs=self.batch_size

        # Each Domain BN
        totalx=[]
        for idx, i in enumerate(uniq_domain[0]):
            bn = self.bns[i]
            totalx.append(bn(x[idx*bs : idx*bs+uniq_domain[1][idx]]))
            
        return torch.cat(totalx), domain
