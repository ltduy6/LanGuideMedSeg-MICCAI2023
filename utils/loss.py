import torch
import torch.nn as nn
import torch.nn.functional as F

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
        

class FeatureFiltrationLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FeatureFiltrationLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = 1e-8

    def forward(self, teacher, student):
        # teacher, student: [B, C, H, W] or [B, L, C]
        
        # 1. Compute Importance Map W_T
        # If input is flattened [B, L, C], view it as spatial if possible or just sum over C.
        # The prompt implies spatial W_T \in R^{H \times W}.
        # If inputs are [B, C, H, W], we sum over C.
        
        if teacher.dim() == 4:
            # [B, C, H, W]
            # Sum absolute activations over channel dimension
            w_t = torch.sum(torch.abs(teacher), dim=1, keepdim=True) # [B, 1, H, W]
        elif teacher.dim() == 3:
            # [B, L, C] -> we assume L corresponds to spatial.
            # We can compute importance per token.
            w_t = torch.sum(torch.abs(teacher), dim=2, keepdim=True) # [B, L, 1]
        else:
            raise ValueError(f"Unsupported input shape: {teacher.shape}")

        # 2. Normalize W_T to [0, 1] per sample
        # Min-Max normalization
        # Flatten W_T for min/max computation per sample
        if teacher.dim() == 4:
            w_t_flat = w_t.view(w_t.size(0), -1)
        else:
            w_t_flat = w_t.view(w_t.size(0), -1)

        w_min = w_t_flat.min(dim=1, keepdim=True)[0]
        w_max = w_t_flat.max(dim=1, keepdim=True)[0]

        # Reshape for broadcasting
        if teacher.dim() == 4:
            # [B, 1] -> [B, 1, 1, 1]
            w_min = w_min.view(w_t.size(0), 1, 1, 1)
            w_max = w_max.view(w_t.size(0), 1, 1, 1)
        else:
            # [B, 1] -> [B, 1, 1]
            w_min = w_min.view(w_t.size(0), 1, 1)
            w_max = w_max.view(w_t.size(0), 1, 1)
        
        w_t_norm = (w_t - w_min) / (w_max - w_min + self.epsilon)
        
        # 3. Compute Loss
        # L_FFD = || W_T * (F_S - F_T) ||_2^2
        diff = student - teacher
        weighted_diff = w_t_norm * diff
        
        # Squared L2 norm
        loss = torch.sum(weighted_diff ** 2, dim=tuple(range(1, weighted_diff.dim()))) 
        # Note: torch.norm(..., p=2)**2 is sum of squares.
        # But wait, formula usually implies mean over elements or sum.
        # ||.||_2^2 is sum of squares of elements.
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

        else:
            return loss


class LogitDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(LogitDistillationLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, teacher_logits, student_logits):
        # teacher_logits, student_logits: [B, 1, H, W] for binary segmentation
        # We need to handle the binary case carefully.
        # usually teacher_logits are raw logits.
        
        # B, 1, H, W -> B, 2, H, W
        # channel 0: -logits, channel 1: logits
        
        s_logits_2ch = torch.cat((-student_logits, student_logits), dim=1)
        t_logits_2ch = torch.cat((-teacher_logits, teacher_logits), dim=1)
        
        # Apply temperature
        s_soft = F.log_softmax(s_logits_2ch / self.temperature, dim=1)
        t_soft = F.softmax(t_logits_2ch / self.temperature, dim=1)
        
        # KL Div
        loss = self.criterion(s_soft, t_soft)
        
        # Scale back
        loss = loss * (self.temperature ** 2)
        
        return loss


class MultiTemperatureKDLoss(nn.Module):
    def __init__(self, temps=[2.0, 3.0, 4.0, 5.0, 6.0], reduction='batchmean'):
        super(MultiTemperatureKDLoss, self).__init__()
        self.temps = temps
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, teacher_logits, student_logits):
        # teacher_logits, student_logits: [B, 1, H, W]
        # returns scalar loss

        total_loss = 0.0

        for t in self.temps:
            # 1. Compute temperature-scaled probabilities
            # p_t_fg = sigmoid(logits_teacher / t)
            # p_s_fg = sigmoid(logits_student / t)
            
            p_t_fg = torch.sigmoid(teacher_logits / t)
            p_s_fg = torch.sigmoid(student_logits / t)
            
            p_t_bg = 1.0 - p_t_fg
            p_s_bg = 1.0 - p_s_fg
            
            # 2. Stack into 2-class distributions: [B, 2, H, W]
            p_t = torch.cat([p_t_bg, p_t_fg], dim=1)
            p_s = torch.cat([p_s_bg, p_s_fg], dim=1)
            
            # 3. Flatten over pixels: [N, 2] where N = B*H*W
            # permute to [B, H, W, 2] then reshape
            pt_flat = p_t.permute(0, 2, 3, 1).reshape(-1, 2)
            ps_flat = p_s.permute(0, 2, 3, 1).reshape(-1, 2)
            
            # 4. Compute pixel-wise KL(teacher || student)
            # KLDivLoss expects input in log-space (log-probabilities) and target as probabilities
            # So apply log to student probs
            
            # p_s might be 0 or 1, so clamp or add epsilon for stability before log?
            # However, sigmoid output is (0, 1) strictly unless large logits.
            # To be safe, we can use log_softmax on the logits constructed from probs?
            # Actually, standard way if we have probabilities `ps_flat` is `ps_flat.log()`.
            # Let's add epsilon for numerical stability.
            
            ps_log = torch.log(ps_flat + 1e-8)
            
            loss_t = self.criterion(ps_log, pt_flat)
            
            total_loss += loss_t
            
        return total_loss
