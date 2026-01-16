from models.baseline_kd import BaselineKDModel
from models.teacher import TeacherModel
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
import torch.nn.functional as F
from .wrapper_mapper import MapperLoss
from utils.loss import FeatureDistillationLoss

class BaselineKDWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(BaselineKDWrapper, self).__init__()
        
        self.model = BaselineKDModel(
            args.bert_type,
            args.vision_type,
            args.project_dim
        )

        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()
        # Initialize FeatureDistillationLoss. Default to p=2 (L_FD2)
        # You can change p=1 for L_FD1. 
        # Ideally this should be configurable via args, but for now fixed or based on request.
        # User requested both, let's assume valid default is needed or check args.
        # The user said "given that z_out_1 is os4...".
        # I'll enable p=2 by default as it's standard L2.
        self.distill_loss_fn = FeatureDistillationLoss(p=2)

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters()
        
        self.lambda_distill_os16 = args.lambda_distill_os16
        self.lambda_distill_os8 = args.lambda_distill_os8
        self.lambda_distill_os4 = args.lambda_distill_os4

        if args.mode == "Training":
            self.teacher_model = TeacherModel(
                args.bert_type,
                args.vision_type,
                args.project_dim
            )
            self.load_teacher_model(args.teacher_model_path)
            for p in self.teacher_model.parameters():
                p.requires_grad = False 
        
    def load_teacher_model(self, path):
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # 1. Try 'state_dict' (PL default) or 'model_state_dict' (Custom)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint # Maybe it's just the state dict
            
            # 2. Fix key prefix if trained with Wrapper
            # TeacherWrapper has self.model, so keys are 'model.xxx'
            # We want to load into self.teacher_model (which is a TeacherModel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v # remove 'model.'
                else:
                    new_state_dict[k] = v
            
            self.teacher_model.load_state_dict(new_state_dict)
            print(f"Loaded teacher model from {path}")
            
        except Exception as e:
            print(f"Failed to load teacher model: {e}")
            raise e
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       
       return self.model.forward(x)


    def shared_step(self,batch,batch_idx):
        x, y = batch
        preds, return_info = self(x)
        main_loss = self.loss_fn(preds,y)

        if self.training:

            with torch.no_grad():
                teacher_preds, teacher_return_info = self.teacher_model(x)
                loss_os16 = self.distill_loss_fn(teacher_return_info['os16'],return_info['os16'])
                loss_os8 = self.distill_loss_fn(teacher_return_info['os8'],return_info['os8'])
                loss_os4 = self.distill_loss_fn(teacher_return_info['os4'],return_info['os4'])

            distill_loss = (
                self.lambda_distill_os16 * loss_os16 + 
                self.lambda_distill_os8 * loss_os8 + 
                self.lambda_distill_os4 * loss_os4
            )
            
            total_loss = main_loss + distill_loss
            
            self.log('train_distill_os16', loss_os16, prog_bar=True)
            self.log('train_distill_os8', loss_os8, prog_bar=True)
            self.log('train_distill_os4', loss_os4, prog_bar=True)
            
            return {
                'loss': total_loss, 
                'preds': preds.detach(),
                'y': y.detach(),
                'main_loss': main_loss.detach()
            }
        else:
            return {
                'loss': main_loss,
                'preds': preds.detach(),
                'y': y.detach(),
                'main_loss': main_loss.detach()
            }

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

    def on_save_checkpoint(self, checkpoint):
        # Remove teacher model from state_dict to save space since it is frozen
        keys_to_remove = [k for k in checkpoint['state_dict'].keys() if k.startswith("teacher_model.")]
        for k in keys_to_remove:
            del checkpoint['state_dict'][k]