from utils.model import LanGuideMedSeg
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from monai.metrics import compute_hausdorff_distance
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime

class HD95Wrapper(nn.Module):
    """Wrapper to make MONAI's compute_hausdorff_distance compatible with nn.ModuleDict"""
    def __init__(self, percentile=95, include_background=False):
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



class LanGuideMedSegWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(LanGuideMedSegWrapper, self).__init__()
        
        # Get training dataset size from args
        train_dataset_size = getattr(args, 'train_dataset_size', None)

        self.model = LanGuideMedSeg(args.bert_type, args.vision_type, args.project_dim, dataset_size=train_dataset_size)
        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex(),"hd95": HD95Wrapper(percentile=95, include_background=False)}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       
       return self.model.forward(x)


    def shared_step(self,batch,batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds,y)
        return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()}    
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)