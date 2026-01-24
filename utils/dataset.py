import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd, RandGaussianNoised)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import open_clip

class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        if mode == 'train':
            self.image_list = self.image_list[:int(0.8*len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8*len(self.image_list)):]
            self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
        else:
            pass   # for mode is 'test'

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'Images',self.image_list[idx].replace('mask_',''))
        gt = os.path.join(self.root_path,'Ground-truths', self.image_list[idx])
        caption = self.caption_list[idx].split(',')[-1].strip()

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']
        gt = torch.where(gt==255,1,0)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                RandRotated(keys=["image","gt"], range_x=[-0.3, 0.3], keep_size=True, mode=['bicubic','nearest'],  prob=0.3),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans

class MosMedPlus(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(MosMedPlus, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'Images',self.image_list[idx])
        gt = os.path.join(self.root_path,'Ground-truths', self.image_list[idx])
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']

        if gt.shape[0] == 3:
            gt = gt.mean(dim=0, keepdim=True)

        gt = torch.where(gt==255,1,0)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                RandRotated(keys=["image","gt"], range_x=[-0.35, 0.35], keep_size=True, mode=['bicubic','nearest'],  prob=0.3),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans
    
class SegmentData(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224], text_length=24, name='QaTa'):

        super(SegmentData, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        if name in ['QaTa', 'MosMedPlus']:
            self.caption_list = [cap.split(',')[-1].strip() for cap in self.caption_list]

        if name in ['QaTa', 'Clinic', 'Kvasir']:
            if mode == 'train':
                self.image_list = self.image_list[:int(0.8*len(self.image_list))]
                self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
            elif mode == 'valid':
                self.image_list = self.image_list[int(0.8*len(self.image_list)):]
                self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
            else:
                pass   # for mode is 'test'

        self.root_path = root_path
        self.image_size = image_size
        self.text_length = text_length
        self.tokenizer_name = tokenizer

        if self.tokenizer_name == 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224':
            self.tokenizer = open_clip.get_tokenizer(self.tokenizer_name).tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'Images',self.image_list[idx].replace('mask_',''))
        gt = os.path.join(self.root_path,'Ground-truths', self.image_list[idx])
        caption = self.caption_list[idx]

        if self.tokenizer_name == 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224':
            token_output = self.tokenizer(
                text=caption,
                max_length=self.text_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                            max_length=self.text_length, 
                                                            truncation=True,
                                                            return_attention_mask=True,
                                                            return_tensors='pt')
            
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']

        if gt.shape[0] == 3:
            gt = gt.mean(dim=0, keepdim=True)
            
        gt = torch.where(gt==255,1,0)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                RandRotated(keys=["image","gt"], range_x=[-0.3, 0.3], keep_size=True, mode=['bicubic','nearest'],  prob=0.3),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])

        return trans