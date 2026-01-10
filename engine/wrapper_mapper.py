import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.model import VisionModel, BERTModel
from utils.mapper import ImageToTextSemanticMapper
from utils.dataset import QaTa, MosMedPlus
from einops import repeat
from pathlib import Path
from einops import rearrange
import datetime
import sys
import numpy as np
import pandas as pd


class MapperLoss(nn.Module):
    """Combined loss function for mapper pretraining"""
    
    def __init__(self, 
                 lambda_token=1.0,
                 lambda_global=0.5,
                 lambda_diversity=0.1,
                 lambda_contrastive=0.2,
                 temperature=0.07):
        super().__init__()
        self.lambda_token = lambda_token
        self.lambda_global = lambda_global
        self.lambda_diversity = lambda_diversity
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
    
    def cosine_token_loss(self, pred, target):
        """Token-level cosine similarity loss"""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 1 - (pred * target).sum(dim=-1).mean()
    
    def global_cosine(self, pred, target):
        """Global cosine similarity loss"""
        pred_global = pred.mean(dim=1)
        target_global = target.mean(dim=1)
        return 1 - F.cosine_similarity(pred_global, target_global).mean()
    
    def diversity_loss(self, tokens):
        """Diversity regularization loss"""
        # tokens: [B, T, C]
        tokens = F.normalize(tokens, dim=-1)
        sim = torch.matmul(tokens, tokens.transpose(-1, -2))  # [B, T, T]
        identity = torch.eye(sim.size(-1), device=sim.device).unsqueeze(0)
        return ((sim - identity)**2).mean()
    
    def contrastive_loss(self, pred, target):
        """Contrastive learning loss"""
        pred = F.normalize(pred.mean(dim=1), dim=-1)
        target = F.normalize(target.mean(dim=1), dim=-1)
        
        logits = pred @ target.T / self.temperature
        labels = torch.arange(pred.size(0), device=pred.device)
        
        return F.cross_entropy(logits, labels)
    
    def forward(self, pred, target):
        """
        pred: [B, 24, 768] - generated visual tokens
        target: [B, 24, 768] - real text tokens
        """
        loss_token = self.cosine_token_loss(pred, target)
        loss_global = self.global_cosine(pred, target)
        loss_diversity = self.diversity_loss(pred)
        loss_contrastive = self.contrastive_loss(pred, target)
        
        total_loss = (
            self.lambda_token * loss_token +
            self.lambda_global * loss_global +
            self.lambda_diversity * loss_diversity +
            self.lambda_contrastive * loss_contrastive
        )
        
        return {
            'loss': total_loss,
            'token_loss': loss_token,
            'global_loss': loss_global,
            'diversity_loss': loss_diversity,
            'contrastive_loss': loss_contrastive
        }


class MapperPretrainer(pl.LightningModule):
    """PyTorch Lightning module for mapper pretraining"""
    
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        
        # Frozen encoders
        self.vision_encoder = VisionModel(args.vision_type, args.project_dim)
        self.text_encoder = BERTModel(args.bert_type, args.project_dim)
        
        # Freeze encoders
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Trainable mapper
        self.mapper = ImageToTextSemanticMapper(
            token_dim=768,
            num_text_tokens=24
        )
        
        # Loss function
        self.criterion = MapperLoss(
            lambda_token=args.lambda_token,
            lambda_global=args.lambda_global,
            lambda_diversity=args.lambda_diversity,
            lambda_contrastive=args.lambda_contrastive,
            temperature=args.temperature
        )

        self.history = {}
        
        self.lr = args.lr
        self.args = args
    
    def forward(self, image, text):
        """Extract features and generate tokens"""
        # Convert grayscale to RGB if needed
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        
        # Extract frozen features
        with torch.no_grad():
            image_output = self.vision_encoder(image)
            image_features = image_output['feature']
            
            text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
            text_embeds = text_output['feature']
        
        # Convert image features to sequence
        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]
            image_features = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features]
        
        # Get visual tokens from stage 4 (os32)
        visual_tokens = image_features[3]  # [B, 49, 768]
        
        # Get text tokens (last layer)
        text_tokens = text_embeds[-1]  # [B, seq_len, 768]
        
        # Take first 24 tokens from text (or pad/truncate)
        if text_tokens.size(1) >= 24:
            text_tokens = text_tokens[:, :24, :]
        else:
            # Pad if needed
            pad_size = 24 - text_tokens.size(1)
            padding = torch.zeros(
                text_tokens.size(0), pad_size, text_tokens.size(2),
                device=text_tokens.device
            )
            text_tokens = torch.cat([text_tokens, padding], dim=1)
        
        # Generate visual tokens through mapper
        generated_visual_tokens = self.mapper(visual_tokens)  # [B, 24, 768]
        
        return generated_visual_tokens, text_tokens
    
    def shared_step(self, batch, batch_idx, stage='train'):
        [image, text], _ = batch
        
        # Forward pass
        pred_tokens, target_tokens = self(image, text)
        
        # Compute losses
        loss_dict = self.criterion(pred_tokens, target_tokens)

        return loss_dict['loss']
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
    
    def shared_step_end(self,outputs,stage):
        return outputs.mean()

    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
    
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
    
    def shared_epoch_end(self,outputs,stage="train"):
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}

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
            self.save_mapper(self.args.mapper_save_path)

    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.mapper.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def save_mapper(self, save_path):
        """Save only the mapper weights"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'mapper_state_dict': self.mapper.state_dict(),
            'args': self.args,
        }, save_path)
        
        print(f"Mapper saved to {save_path}")
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)

    def get_history(self):
        return pd.DataFrame(self.history.values()) 