import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel



class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        # self.project_head = nn.Linear(768, project_dim)
        # self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        # embeds = output['pooler_output'].squeeze()
        # project = self.project_head(embeds)

        return {"feature":output['hidden_states']}

class DINOv2VisionModel(nn.Module):
    """DINOv2-specific vision encoder"""
    
    def __init__(self, vision_type="facebook/dinov2-base", project_dim=768):
        super(DINOv2VisionModel, self).__init__()
        
        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True)
        self.vision_type = vision_type
        
        # DINOv2-base configurations
        self.patch_size = 14  # DINOv2 uses 14x14 patches
        self.hidden_size = 768  # DINOv2-base hidden dimension
        self.num_layers = 12  # DINOv2-base has 12 transformer blocks
        
        # Check if model has register tokens
        self.has_registers = 'reg' in vision_type.lower()
        self.num_register_tokens = 4 if self.has_registers else 0
        
        # For 224x224 input with patch_size=14, we get 16x16 = 256 patches
        self.num_patches_per_side = 224 // self.patch_size  # 16
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        output = self.model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        hidden_states = output['hidden_states']  # Tuple of (13,) each [B, 257, 768] or [B, 261, 768] with registers
        
        processed_features = []
        
        for hidden_state in hidden_states:
            # Remove CLS token and register tokens
            tokens_to_remove = 1 + self.num_register_tokens  # CLS + register tokens
            patch_embeddings = hidden_state[:, tokens_to_remove:, :]  # [B, 256, 768]
            
            # Reshape to spatial format [B, 768, 16, 16]
            spatial_features = rearrange(
                patch_embeddings, 
                'b (h w) c -> b c h w', 
                h=self.num_patches_per_side, 
                w=self.num_patches_per_side
            )
            
            processed_features.append(spatial_features)
        
        return {"feature": processed_features}


class LanGuideMedSeg_DINOv2(nn.Module):
    """LanGuideMedSeg with DINOv2-base encoder"""
    
    def __init__(self, bert_type, vision_type="facebook/dinov2-base", project_dim=768):
        super(LanGuideMedSeg_DINOv2, self).__init__()

        # DINOv2 encoder
        self.encoder = DINOv2VisionModel(vision_type, project_dim)
        # self.text_encoder = BERTModel(bert_type, project_dim)
        
        # DINOv2-base produces 16x16 feature maps from 224x224 input
        # We need to create hierarchical features for the decoder
        
        base_dim = 768  # DINOv2-base hidden dimension
        
        # Multi-scale feature pyramid from DINOv2 single-scale output
        # We'll use different transformer layers and upsample them
        self.feature_pyramid = nn.ModuleDict({
            'layer_9': nn.Sequential(  # Early layer -> coarser features
                nn.Conv2d(base_dim, 768, 1),
                nn.BatchNorm2d(768),
                nn.GELU()
            ),  # 16x16 -> keep at 768 dim (os32 equivalent)
            
            'layer_10': nn.Sequential(  # Mid layer
                nn.Conv2d(base_dim, 384, 1),
                nn.BatchNorm2d(384),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),  # 16x16 -> 32x32, 384 dim (os16 equivalent)
            
            'layer_11': nn.Sequential(  # Late layer
                nn.Conv2d(base_dim, 192, 1),
                nn.BatchNorm2d(192),
                nn.GELU(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            ),  # 16x16 -> 64x64, 192 dim (os8 equivalent)
            
            'layer_12': nn.Sequential(  # Final layer -> finest features
                nn.Conv2d(base_dim, 96, 1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
            ),  # 16x16 -> 128x128, 96 dim (os4 equivalent)
        })
        
        # Adjusted spatial dimensions to match DINOv2 output scales
        self.spatial_dim = [16, 32, 64, 128]  # Spatial sizes after feature pyramid
        feature_dim = [768, 384, 192, 96]
        
        # Decoders (same as original LanGuideMedSeg but with adjusted spatial dims)
        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        image, text, gt = data
        
        if image.shape[1] == 1:   
            image = repeat(image, 'b 1 h w -> b c h w', c=3)

        # DINOv2 encoder - get all hidden states
        image_output = self.encoder(image)
        dinov2_features = image_output['feature']  # List of 13 layers, each [B, 768, 16, 16]
        
        # Text encoder
        # text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        # text_embeds = text_output['feature']
        
        # Select specific layers for multi-scale features
        # Use layers: 9, 10, 11, 12 (later layers have more semantic info)
        selected_layers = [dinov2_features[9], dinov2_features[10], 
                          dinov2_features[11], dinov2_features[12]]
        
        # Create multi-scale feature pyramid
        image_features = []
        for idx, (layer_name, layer_feature) in enumerate(zip(
            ['layer_9', 'layer_10', 'layer_11', 'layer_12'], selected_layers)):
            
            # Apply feature pyramid transformation
            transformed_feature = self.feature_pyramid[layer_name](layer_feature)
            # [B, 768, 16, 16], [B, 384, 32, 32], [B, 192, 64, 64], [B, 96, 128, 128]
            
            # Convert to sequence format for GuideDecoder
            seq_feature = rearrange(transformed_feature, 'b c h w -> b (h w) c')
            image_features.append(seq_feature)
        
        # Decoder pathway
        os32 = image_features[0]  # [B, 256, 768] (16x16)
        os16 = self.decoder16(os32, image_features[1], None)  # -> [B, 1024, 384] (32x32)
        os8 = self.decoder8(os16, image_features[2], None)    # -> [B, 4096, 192] (64x64)
        os4 = self.decoder4(os8, image_features[3], None)     # -> [B, 16384, 96] (128x128)

        # Back to spatial format
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)  # -> [B, 24, 224, 224]

        out = self.out(os1).sigmoid()  # -> [B, 1, 224, 224]

        return out


class LanGuideMedSeg_DINOv2_Adaptive(nn.Module):
    """
    Adaptive version that can work with different DINOv2 variants
    """
    
    def __init__(self, bert_type, vision_type="facebook/dinov2-base", project_dim=768):
        super(LanGuideMedSeg_DINOv2_Adaptive, self).__init__()

        self.encoder = DINOv2VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        
        # Determine base dimension based on DINOv2 variant
        if 'small' in vision_type:
            base_dim = 384
        elif 'large' in vision_type:
            base_dim = 1024
        elif 'giant' in vision_type:
            base_dim = 1536
        else:  # base
            base_dim = 768
            
        self.base_dim = base_dim
        
        # Adaptive feature pyramid
        self.feature_pyramid = nn.ModuleDict({
            'scale_1': self._make_pyramid_layer(base_dim, 768, scale=1),
            'scale_2': self._make_pyramid_layer(base_dim, 384, scale=2),
            'scale_3': self._make_pyramid_layer(base_dim, 192, scale=4),
            'scale_4': self._make_pyramid_layer(base_dim, 96, scale=8),
        })
        
        self.spatial_dim = [16, 32, 64, 128]
        feature_dim = [768, 384, 192, 96]
        
        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
    
    def _make_pyramid_layer(self, in_dim, out_dim, scale=1):
        """Create a pyramid layer with optional upsampling"""
        layers = [
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        ]
        if scale > 1:
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def forward(self, data):
        image, text, gt = data
        
        if image.shape[1] == 1:   
            image = repeat(image, 'b 1 h w -> b c h w', c=3)

        image_output = self.encoder(image)
        dinov2_features = image_output['feature']
        
        text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_embeds = text_output['feature']
        
        # Use last 4 layers for feature pyramid
        num_layers = len(dinov2_features)
        selected_layers = [dinov2_features[num_layers-4], dinov2_features[num_layers-3],
                          dinov2_features[num_layers-2], dinov2_features[num_layers-1]]
        
        # Create multi-scale features
        image_features = []
        for idx, (scale_name, layer_feature) in enumerate(zip(
            ['scale_1', 'scale_2', 'scale_3', 'scale_4'], selected_layers)):
            
            transformed = self.feature_pyramid[scale_name](layer_feature)
            seq_feature = rearrange(transformed, 'b c h w -> b (h w) c')
            image_features.append(seq_feature)
        
        # Decoder
        os32 = image_features[0]
        os16 = self.decoder16(os32, image_features[1], text_embeds[-1])
        os8 = self.decoder8(os16, image_features[2], text_embeds[-1])
        os4 = self.decoder4(os8, image_features[3], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()

        return out


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        # self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text, gt = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features = image_output['feature']
        # text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        # text_embeds, text_project = text_output['feature'],text_output['project']

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

        return out

    
