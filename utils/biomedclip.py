import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
import open_clip

class BiomedCLIPEncoder(nn.Module):
    """
    Unified BiomedCLIP Encoder for both vision and text.
    Uses a single shared model instance.
    """
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", project_dim=512):
        super().__init__()
        
        # Load BiomedCLIP ONCE - shared for both vision and text
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Extract vision and text encoders (they reference the same base model)
        self.visual = self.model.visual
        self.text_encoder = self.model.text
        
        # ViT-B/16 configuration
        self.patch_size = 16
        self.hidden_size = 768
        self.num_layers = 12
        self.image_size = 224
        
        # Optional: Additional projection heads for task-specific features
        self.vision_project_head = nn.Sequential(
            nn.Linear(self.hidden_size, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        
        self.text_project_head = nn.Sequential(
            nn.Linear(self.hidden_size, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        
    def encode_image(self, x):
        """
        Encode images and return multi-scale features
        x: [B, 3, 224, 224]
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
        
        # Final processing for CLS token
        cls_final = self.visual.ln_post(x[:, 0, :])
        
        # Use BiomedCLIP's original projection
        if self.visual.proj is not None:
            clip_features = cls_final @ self.visual.proj
        else:
            clip_features = cls_final
        
        # Additional task-specific projection
        project_embed = self.vision_project_head(cls_final)
        
        # Process hidden states to spatial format
        processed_features = []
        for hidden_state in hidden_states:
            patch_tokens = hidden_state[:, 1:, :]  # Remove CLS token
            spatial_features = rearrange(
                patch_tokens,
                'b (h w) c -> b c h w',
                h=14, w=14
            )
            processed_features.append(spatial_features)
        
        return {
            "feature": processed_features,      # Multi-scale spatial features
            "project": project_embed,           # Task-specific projection
            "clip_features": clip_features      # Original CLIP features
        }
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text and return features
        input_ids: [B, L]
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
        
        # Final layer norm
        x = x.permute(1, 0, 2)
        x = self.text_encoder.ln_final(x)
        
        # Extract features from EOT token
        eot_indices = input_ids.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_indices]
        
        # Use BiomedCLIP's original projection
        if self.text_encoder.text_projection is not None:
            clip_features = text_features @ self.text_encoder.text_projection
        else:
            clip_features = text_features
        
        # Additional task-specific projection
        project_embed = self.text_project_head(text_features)
        
        return {
            "feature": hidden_states,           # All layer features
            "project": project_embed,           # Task-specific projection
            "clip_features": clip_features      # Original CLIP features
        }


class LanGuideMedSeg_BiomedCLIP(nn.Module):
    """
    LanGuideMedSeg using single shared BiomedCLIP model.
    More memory efficient and maintains pre-trained alignment.
    """
    
    def __init__(self, 
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 project_dim=512):
        super().__init__()
        
        # Single shared BiomedCLIP encoder
        self.biomedclip = BiomedCLIPEncoder(model_name, project_dim)
        
        base_dim = 768
        
        # Feature pyramid
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
        
        # Decoders
        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        image, text = data
        
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        
        # Encode with shared BiomedCLIP
        image_output = self.biomedclip.encode_image(image)
        vision_features = image_output['feature']
        
        text_output = self.biomedclip.encode_text(text['input_ids'], text.get('attention_mask'))
        text_embeds = text_output['feature']
        
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
        
        return out


class LanGuideMedSeg_BiomedCLIP_WithContrastive(nn.Module):
    """
    BiomedCLIP version with contrastive learning.
    Uses single shared model for memory efficiency.
    """
    
    def __init__(self, 
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 project_dim=512):
        super().__init__()
        
        # Single shared BiomedCLIP encoder
        self.biomedclip = BiomedCLIPEncoder(model_name, project_dim)
        
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
        
        # Encode with shared BiomedCLIP
        image_output = self.biomedclip.encode_image(image)
        vision_features = image_output['feature']
        image_clip = image_output['clip_features']      # Original CLIP features
        image_project = image_output['project']         # Task-specific projection
        
        text_output = self.biomedclip.encode_text(text['input_ids'], text.get('attention_mask'))
        text_embeds = text_output['feature']
        text_clip = text_output['clip_features']        # Original CLIP features
        text_project = text_output['project']           # Task-specific projection
        
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
            # Can return either CLIP features (pre-trained) or task-specific projections
            # For contrastive loss, use CLIP features to maintain pre-trained alignment
            return out, image_clip, text_clip
        
        return out