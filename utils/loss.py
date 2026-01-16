import torch
import torch.nn as nn

class FeatureDistillationLoss(nn.Module):
    def __init__(self, p=2, reduction='mean'):
        super(FeatureDistillationLoss, self).__init__()
        self.p = p
        self.reduction = reduction
        self.epsilon = 1e-8

    def forward(self, teacher, student):
        # Flatten spatial and channel dimensions for normalization
        # We want to align the features, treating them as a vector (or feature map)
        # Based on formula: normalize the whole feature map z_out
        
        s_flat = student.reshape(student.size(0), -1)
        t_flat = teacher.reshape(teacher.size(0), -1)
        
        # Compute norms per sample
        s_norm = torch.norm(s_flat, p=self.p, dim=1, keepdim=True) + self.epsilon
        t_norm = torch.norm(t_flat, p=self.p, dim=1, keepdim=True) + self.epsilon
        
        # Normalize
        s_normalized = s_flat / s_norm
        t_normalized = t_flat / t_norm
        
        # Compute distance
        # Formula: || s/|s| - t/|t| ||_p
        diff = s_normalized - t_normalized
        dist = torch.norm(diff, p=self.p, dim=1)
        
        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        else:
            return dist
