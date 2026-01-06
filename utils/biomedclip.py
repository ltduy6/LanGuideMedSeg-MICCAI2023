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
    Properly handles TimmModel wrapper for vision and HFTextEncoder (BERT) for text.
    """
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", project_dim=512):
        super().__init__()
        
        # Load BiomedCLIP ONCE - shared for both vision and text
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Extract vision and text encoders
        self.visual = self.model.visual  # TimmModel wrapper
        self.text_encoder = self.model.text  # HFTextEncoder (BERT)
        
        # Access the actual ViT through trunk
        self.vit = self.visual.trunk
        
        # Access BERT model
        self.bert = self.text_encoder.transformer
        
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
        
        # Access ViT trunk from TimmModel
        vit = self.vit
        
        # Patch embedding using timm's patch_embed
        x = vit.patch_embed(x)  # [B, 196, 768]
        
        # Add cls token and positional embedding
        cls_token = vit.cls_token.expand(batch_size, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = vit.pos_drop(x + vit.pos_embed)
        
        # Collect features from all transformer blocks
        hidden_states = []
        for block in vit.blocks:
            x = block(x)
            hidden_states.append(x)
        
        # Final layer norm
        x = vit.norm(x)
        
        # Extract CLS token
        cls_features = x[:, 0]  # [B, 768]
        
        # Use BiomedCLIP's projection from TimmModel.head.proj
        if hasattr(self.visual.head, 'proj'):
            clip_features = self.visual.head.proj(cls_features)  # [B, 512]
        else:
            clip_features = cls_features
        
        # Additional task-specific projection
        project_embed = self.vision_project_head(cls_features)
        
        # Process hidden states to spatial format
        processed_features = []
        for hidden_state in hidden_states:
            # Remove CLS token: [B, 197, 768] -> [B, 196, 768]
            patch_tokens = hidden_state[:, 1:, :]
            
            # Reshape to spatial [B, 768, 14, 14]
            spatial_features = rearrange(
                patch_tokens,
                'b (h w) c -> b c h w',
                h=14, w=14
            )
            processed_features.append(spatial_features)
        
        return {
            "feature": processed_features,      # Multi-scale spatial features [12 x [B, 768, 14, 14]]
            "project": project_embed,           # Task-specific projection [B, project_dim]
            "clip_features": clip_features      # Original CLIP features [B, 512]
        }
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text using BERT and return features
        input_ids: [B, L] - tokenized text
        attention_mask: [B, L] - attention mask (optional)
        """
        # BERT expects attention_mask
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()  # Create mask from padding
        
        # Get BERT embeddings
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=None
        )
        
        # Collect features from all BERT layers
        hidden_states = []
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        
        hidden = embedding_output
        for layer in self.bert.encoder.layer:
            hidden = layer(hidden, extended_attention_mask)[0]
            hidden_states.append(hidden)
        
        # Use pooler to get CLS token representation
        pooled_output = self.text_encoder.pooler(hidden)  # [B, 768]
        
        # Use BiomedCLIP's text projection
        if hasattr(self.text_encoder, 'proj'):
            clip_features = self.text_encoder.proj(pooled_output)  # [B, 512]
        else:
            clip_features = pooled_output
        
        # Additional task-specific projection
        project_embed = self.text_project_head(pooled_output)
        
        return {
            "feature": hidden_states,           # All layer features [12 x [B, L, 768]]
            "project": project_embed,           # Task-specific projection [B, project_dim]
            "clip_features": clip_features      # Original CLIP features [B, 512]
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
        
        # Feature pyramid to create multi-scale features from ViT
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
        
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        
        # Encode with shared BiomedCLIP
        image_output = self.biomedclip.encode_image(image)
        vision_features = image_output['feature']  # List of 12 layers [B, 768, 14, 14]
        
        text_output = self.biomedclip.encode_text(
            text['input_ids'], 
            text.get('attention_mask')
        )
        text_embeds = text_output['feature']  # List of 12 BERT layers [B, L, 768]
        
        # Multi-scale features from last 4 ViT blocks (layers 9, 10, 11, 12)
        selected_layers = [vision_features[8], vision_features[9],
                          vision_features[10], vision_features[11]]
        
        image_features = []
        for idx, (layer_name, layer_feature) in enumerate(zip(
            ['layer_9', 'layer_10', 'layer_11', 'layer_12'], selected_layers)):
            
            # Apply feature pyramid transformation
            transformed = self.feature_pyramid[layer_name](layer_feature)
            # Convert to sequence format [B, H*W, C]
            seq_feature = rearrange(transformed, 'b c h w -> b (h w) c')
            image_features.append(seq_feature)
        
        # Use last BERT layer for guidance
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
        image_clip = image_output['clip_features']      # [B, 512] - Original CLIP features
        image_project = image_output['project']         # [B, project_dim] - Task-specific
        
        text_output = self.biomedclip.encode_text(
            text['input_ids'], 
            text.get('attention_mask')
        )
        text_embeds = text_output['feature']
        text_clip = text_output['clip_features']        # [B, 512] - Original CLIP features
        text_project = text_output['project']           # [B, project_dim] - Task-specific
        
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
            # Return original CLIP features to maintain pre-trained alignment
            return out, image_clip, text_clip
        
        return out