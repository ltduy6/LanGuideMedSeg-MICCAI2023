import torch
import torch.nn as nn
from einops import rearrange, repeat
from utils.layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from .bert import BERTModel
from .vision import VisionModel

class BaselineKDModel(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=718):

        super(BaselineKDModel, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],24)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],24)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        os32 = image_features[3]

        os16 = self.decoder16(os32,image_features[2], None)
        os8 = self.decoder8(os16,image_features[1], None)
        os4 = self.decoder4(os8,image_features[0], None)
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return_info = {
            'os16': os16,
            'os8': os8,
            'os4': os4,
        }

        return out, return_info
    

    