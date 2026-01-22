from monai.metrics import compute_hausdorff_distance
import torch
import torch.nn as nn

class HD95Wrapper(nn.Module):
    """Wrapper to make MONAI's compute_hausdorff_distance compatible with nn.ModuleDict"""
    def __init__(self, percentile=95, include_background=True):
        super().__init__()
        self.percentile = percentile
        self.include_background = include_background
        self.accumulated_values = []
    
    def __call__(self, preds, target):
        try:
            # Ensure tensors are detached and on CPU
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu()
            if isinstance(target, torch.Tensor):
                target = target.detach().cpu()
            
            # Apply sigmoid and binarize predictions if they contain logits
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            
            # Binarize predictions and ensure target is binary
            preds_binary = (preds > 0.5).float()
            target_binary = target.float()
            
            # Ensure correct shape: (batch, channel, height, width)
            if len(preds_binary.shape) == 3:  # (batch, height, width)
                preds_binary = preds_binary.unsqueeze(1)  # Add channel dimension
            if len(target_binary.shape) == 3:  # (batch, height, width)
                target_binary = target_binary.unsqueeze(1)  # Add channel dimension
            
            # Convert to numpy for MONAI
            preds_np = preds_binary.numpy()
            target_np = target_binary.numpy()
            
            # Compute HD95 for each sample in the batch
            batch_size = preds_np.shape[0]
            hd_values = []
            
            for i in range(batch_size):
                pred_sample = preds_np[i:i+1]  # Keep batch dimension
                target_sample = target_np[i:i+1]  # Keep batch dimension
                
                # Skip if either prediction or target is empty
                if pred_sample.sum() == 0 and target_sample.sum() == 0:
                    hd_values.append(0.0)  # Perfect match for empty masks
                elif pred_sample.sum() == 0 or target_sample.sum() == 0:
                    hd_values.append(100.0)  # Large value for no overlap
                else:
                    try:
                        result = compute_hausdorff_distance(
                            pred_sample, target_sample, 
                            include_background=self.include_background, 
                            percentile=self.percentile
                        )
                        hd_values.append(float(result))
                    except Exception as e:
                        print(f"HD95 computation error: {e}")
                        hd_values.append(100.0)  # Default large value
            
            # Return mean HD95 for the batch
            mean_hd = sum(hd_values) / len(hd_values)
            self.accumulated_values.append(mean_hd)
            
            return torch.tensor(mean_hd, dtype=torch.float32)
            
        except Exception as e:
            print(f"HD95Wrapper error: {e}")
            return torch.tensor(100.0, dtype=torch.float32)  # Return large value on error
    
    def reset(self):
        self.accumulated_values = []
    
    def compute(self):
        if len(self.accumulated_values) == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        return torch.tensor(sum(self.accumulated_values) / len(self.accumulated_values), dtype=torch.float32)