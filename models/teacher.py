import torch
import torch.nn as nn
from einops import rearrange, repeat
from utils.layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from .bert import BERTModel
from .vision import VisionModel

class TeacherModel(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=718):

        super(TeacherModel, self).__init__()

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
        text_embeds = text_output['feature']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        os32 = image_features[3]

        os16, refined_os32 = self.decoder16(os32,image_features[2], text_embeds)
        os8, refined_os16 = self.decoder8(os16,image_features[1], text_embeds)
        os4, refined_os8 = self.decoder4(os8,image_features[0], text_embeds)
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        logits = self.out(os1)
        out = logits.sigmoid()

        return_info = {
            'os16': os16,
            'os8': os8,
            'os4': os4,
            'refined_os32': refined_os32,
            'refined_os16': refined_os16,
            'refined_os8': refined_os8,
            'logits': logits,
        }

        return out, return_info
    

    