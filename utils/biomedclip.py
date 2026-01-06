import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
import open_clip

class BiomedCLIPVisionModel(nn.Module):
    """BiomedCLIP Vision Encoder"""
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", project_dim=512):
        super().__init__()
        
        # Load BiomedCLIP
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.visual = self.model.visual  # Extract vision encoder
        
        # ViT-B/16 configuration
        self.patch_size = 16
        self.hidden_size = 768
        self.num_layers = 12
        self.image_size = 224
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 196
        
        # Projection head
        self.project_head = nn.Sequential(
            nn.Linear(self.hidden_size, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        
    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        Returns: Multi-scale features from transformer layers
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.visual.conv1(x)  # [B, 768, 14, 14]
        x = x.reshape(batch_size, self.hidden_size, -1)  # [B, 768, 196]
        x = x.permute(0, 2, 1)  # [B, 196, 768]
        
        # Add CLS token and positional embedding
        cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            batch_size, 1, self.hidden_size, dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # [B, 197, 768]
        x = x + self.visual.positional_embedding.to(x.dtype)
        
        # Layer norm
        x = self.visual.ln_pre(x)
        
        # Collect features from all transformer blocks
        hidden_states = []
        for block in self.visual.transformer.resblocks:
            x = block(x)
            hidden_states.append(x)
        
        # Final layer norm
        x = self.visual.ln_post(x[:, 0, :])  # CLS token
        
        # Project for contrastive learning
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        
        project_embed = self.project_head(x)
        
        # Process hidden states to spatial format
        processed_features = []
        for hidden_state in hidden_states:
            # Remove CLS token
            patch_tokens = hidden_state[:, 1:, :]  # [B, 196, 768]
            
            # Reshape to spatial [B, 768, 14, 14]
            spatial_features = rearrange(
                patch_tokens,
                'b (h w) c -> b c h w',
                h=14, w=14
            )
            processed_features.append(spatial_features)
        
        return {"feature": processed_features, "project": project_embed}


class BiomedCLIPTextModel(nn.Module):
    """BiomedCLIP Text Encoder"""
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", project_dim=512):
        super().__init__()
        
        # Load BiomedCLIP
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_encoder = self.model.text
        
        self.hidden_size = 768
        
        # Projection head
        self.project_head = nn.Sequential(
            nn.Linear(self.hidden_size, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: tokenized text [B, L]
        Returns: Text features from all layers
        """
        # BiomedCLIP text encoding
        x = self.text_encoder.token_embedding(input_ids)
        x = x + self.text_encoder.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Collect features from all transformer blocks
        hidden_states = []
        for block in self.text_encoder.transformer.resblocks:
            x = block(x)
            hidden_states.append(x.permute(1, 0, 2))  # LND -> NLD
        
        # Final layer norm and projection
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_encoder.ln_final(x)
        
        # Take features from EOT token (end of text)
        # Use argmax to find EOT position
        eot_indices = input_ids.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_indices]
        
        # Project
        if self.text_encoder.text_projection is not None:
            text_features = text_features @ self.text_encoder.text_projection
        
        project_embed = self.project_head(text_features)
        
        return {"feature": hidden_states, "project": project_embed}
    
class LanGuideMedSeg_BiomedCLIP(nn.Module):
    """
    LanGuideMedSeg using BiomedCLIP for both vision and text encoding.
    Leverages pre-trained medical vision-language alignment.
    """
    
    def __init__(self, 
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 project_dim=512):
        super().__init__()
        
        # BiomedCLIP encoders (shared model)
        self.encoder = BiomedCLIPVisionModel(model_name, project_dim)
        self.text_encoder = BiomedCLIPTextModel(model_name, project_dim)
        
        # Multi-scale feature pyramid from ViT features
        # ViT-B/16 outputs 14x14 spatial features
        base_dim = 768
        
        self.feature_pyramid = nn.ModuleDict({
            'layer_9': nn.Sequential(
                nn.Conv2d(base_dim, 768, 1),
                nn.BatchNorm2d(768),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((7, 7))  # 14x14 -> 7x7
            ),
            
            'layer_10': nn.Sequential(
                nn.Conv2d(base_dim, 384, 1),
                nn.BatchNorm2d(384),
                nn.GELU(),
                nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)  # Keep 14x14
            ),
            
            'layer_11': nn.Sequential(
                nn.Conv2d(base_dim, 192, 1),
                nn.BatchNorm2d(192),
                nn.GELU(),
                nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)  # 14x14 -> 28x28
            ),
            
            'layer_12': nn.Sequential(
                nn.Conv2d(base_dim, 96, 1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False)  # 14x14 -> 56x56
            ),
        })
        
        # Spatial dimensions and feature dimensions
        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96]
        
        # Decoders with text guidance
        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        image, text = data
        
        # Convert grayscale to RGB
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        
        # Vision features from BiomedCLIP
        image_output = self.encoder(image)
        vision_features = image_output['feature']  # List of 12 layers [B, 768, 14, 14]
        image_project = image_output['project']    # [B, project_dim]
        
        # Text features from BiomedCLIP
        text_output = self.text_encoder(text['input_ids'], text.get('attention_mask'))
        text_embeds = text_output['feature']       # List of text layer features
        text_project = text_output['project']      # [B, project_dim]
        
        # Select layers for multi-scale pyramid (use last 4 layers)
        selected_layers = [vision_features[8], vision_features[9], 
                          vision_features[10], vision_features[11]]
        
        # Create multi-scale features
        image_features = []
        for idx, (layer_name, layer_feature) in enumerate(zip(
            ['layer_9', 'layer_10', 'layer_11', 'layer_12'], selected_layers)):
            
            # Apply feature pyramid transformation
            transformed = self.feature_pyramid[layer_name](layer_feature)
            
            # Convert to sequence format [B, H*W, C]
            seq_feature = rearrange(transformed, 'b c h w -> b (h w) c')
            image_features.append(seq_feature)
        
        # Use last text layer for guidance
        text_guidance = text_embeds[-1]  # [B, L, 768]
        
        # Decoder pathway with text guidance
        os32 = image_features[0]  # [B, 49, 768] (7x7)
        os16 = self.decoder16(os32, image_features[1], text_guidance)  # [B, 196, 384] (14x14)
        os8 = self.decoder8(os16, image_features[2], text_guidance)    # [B, 784, 192] (28x28)
        os4 = self.decoder4(os8, image_features[3], text_guidance)     # [B, 3136, 96] (56x56)
        
        # Reshape and upsample
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)  # [B, 24, 224, 224]
        
        # Final output
        out = self.out(os1).sigmoid()  # [B, 1, 224, 224]
        
        return out


class LanGuideMedSeg_BiomedCLIP_WithContrastive(nn.Module):
    """
    BiomedCLIP version with contrastive learning support.
    Returns embeddings for vision-text alignment loss.
    """
    
    def __init__(self, 
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 project_dim=512):
        super().__init__()
        
        self.encoder = BiomedCLIPVisionModel(model_name, project_dim)
        self.text_encoder = BiomedCLIPTextModel(model_name, project_dim)
        
        base_dim = 768
        
        self.feature_pyramid = nn.ModuleDict({
            'layer_9': nn.Sequential(
                nn.Conv2d(base_dim, 768, 1),
                nn.BatchNorm2d(768),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((7, 7))
            ),
            'layer_10': nn.Sequential(
                nn.Conv2d(base_dim, 384, 1),
                nn.BatchNorm2d(384),
                nn.GELU(),
                nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)
            ),
            'layer_11': nn.Sequential(
                nn.Conv2d(base_dim, 192, 1),
                nn.BatchNorm2d(192),
                nn.GELU(),
                nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)
            ),
            'layer_12': nn.Sequential(
                nn.Conv2d(base_dim, 96, 1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False)
            ),
        })
        
        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96]
        
        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data, return_embeddings=False):
        image, text = data
        
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        
        # Encode
        image_output = self.encoder(image)
        vision_features = image_output['feature']
        image_project = image_output['project']
        
        text_output = self.text_encoder(text['input_ids'], text.get('attention_mask'))
        text_embeds = text_output['feature']
        text_project = text_output['project']
        
        # Multi-scale features
        selected_layers = [vision_features[8], vision_features[9],
                          vision_features[10], vision_features[11]]
        
        image_features = []
        for idx, (layer_name, layer_feature) in enumerate(zip(
            ['layer_9', 'layer_10', 'layer_11', 'layer_12'], selected_layers)):
            
            transformed = self.feature_pyramid[layer_name](layer_feature)
            seq_feature = rearrange(transformed, 'b c h w -> b (h w) c')
            image_features.append(seq_feature)
        
        text_guidance = text_embeds[-1]
        
        # Decode
        os32 = image_features[0]
        os16 = self.decoder16(os32, image_features[1], text_guidance)
        os8 = self.decoder8(os16, image_features[2], text_guidance)
        os4 = self.decoder4(os8, image_features[3], text_guidance)
        
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()
        
        if return_embeddings:
            return out, image_project, text_project
        
        return out