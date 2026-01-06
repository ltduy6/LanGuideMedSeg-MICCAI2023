# === Standard Library ===
from typing import List, Optional

# === Third-party Libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import repeat, rearrange
from tqdm import tqdm

# === BiomedCLIP Utilities ===
from open_clip import create_model_from_pretrained, get_tokenizer

# === Segmentation Components ===
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample


class BiomedCLIP(nn.Module):
    """
    BiomedCLIP wrapper for image-text representation learning and contrastive pretraining.
    """

    def __init__(self, args):
        """
        Initialize BiomedCLIP model and tokenizer.

        Args:
            args (Namespace): Configuration object with model and training settings.
        """
        super().__init__()
        self.args = args
        self.temperature = 0.1
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        # Projection + Classification Head for Image-Text Matching
        feature_dim = 512
        self.itm_head = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2)
        )

    def encode_image(self, image: torch.Tensor, preprocess: bool = False) -> torch.Tensor:
        """
        Encodes image using visual encoder.

        Args:
            image (Tensor or str): Input tensor [B, C, H, W] or image path.
            preprocess (bool): Whether to apply preprocessing.

        Returns:
            Tensor: Encoded image features.
        """
        if preprocess:
            image = self.preprocess(Image.open(image)).unsqueeze(0)

        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b 3 h w')  # Convert grayscale to RGB

        return self.model.encode_image(image.to(self.args.device))

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Tokenizes and encodes text using BERT encoder.

        Args:
            text (List[str]): List of captions or textual descriptions.

        Returns:
            Tensor: Encoded text features.
        """
        tokens = self.tokenizer(text)
        return self.model.encode_text(tokens.to(self.args.device))

    def encode_image_feature(self, image: torch.Tensor, preprocess: bool = False) -> torch.Tensor:
        """
        Extracts intermediate visual features (before projection).

        Returns:
            Tensor: [B, N, C] features.
        """
        if preprocess:
            image = self.preprocess(Image.open(image)).unsqueeze(0)
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b 3 h w')
        return self.model.visual.forward_feature(image.to(self.args.device))

    def encode_text_feature(self, text: List[str]) -> torch.Tensor:
        """
        Extracts intermediate text features (BERT transformer hidden states).

        Returns:
            Tensor: [B, L, C] features.
        """
        tokens = self.tokenizer(text)
        return self.model.text_encoder.forward_feature(tokens.to(self.args.device))

    def contrastive_loss(
        self, image: torch.Tensor, text: List[str], pseudolabel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes contrastive and image-text matching (ITM) losses.

        Returns:
            Tensor: Combined loss value.
        """
        B = image.size(0)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # === Contrastive Loss ===
        logits = torch.matmul(image_features, text_features.T) / self.temperature

        if pseudolabel is not None:
            label_sim = (pseudolabel.unsqueeze(1) == pseudolabel.unsqueeze(0)).all(dim=-1).float()
            contrastive_targets = label_sim.to(logits.device)
        else:
            contrastive_targets = torch.arange(B, device=logits.device)

        loss_img = F.cross_entropy(logits, contrastive_targets)
        loss_txt = F.cross_entropy(logits.T, contrastive_targets)
        contrastive_loss = (loss_img + loss_txt) / 2

        # === Image-Text Matching (ITM) Loss ===
        pos_pairs = torch.cat([image_features, text_features], dim=-1)
        neg_text = text_features[torch.randperm(B)]
        neg_pairs = torch.cat([image_features, neg_text], dim=-1)

        itm_input = torch.cat([pos_pairs, neg_pairs], dim=0)  # [2B, 2D]
        itm_labels = torch.cat([
            torch.ones(B, dtype=torch.long),
            torch.zeros(B, dtype=torch.long)
        ]).to(image.device)

        itm_logits = self.itm_head(itm_input)
        itm_loss = F.cross_entropy(itm_logits, itm_labels)

        return contrastive_loss + itm_loss

    def contrastive_learning(self, data, epochs: int = 5, lr: float = 1e-4):
        """
        Performs contrastive pretraining with BiomedCLIP.

        Args:
            data (DataLoader): Iterable with dicts: {image, text, pseudolabel}.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = AdamW(
            list(self.model.visual.head.parameters()) +
            list(self.model.text.proj.parameters()),
            lr=lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                image = batch["image"]
                text = batch["text"]
                pseudolabel = batch["pseudolabel"]["position"]

                optimizer.zero_grad()
                loss = self.contrastive_loss(image, text, pseudolabel=pseudolabel)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            print(f"[Epoch {epoch+1}] Contrastive Pretrain Loss: {total_loss / len(data):.4f}")


# ============================================================================
# Segmentation Models
# ============================================================================

class BiomedCLIPEncoder(nn.Module):
    """
    Unified BiomedCLIP Encoder for extracting multi-scale features.
    """
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 
                 project_dim=512):
        super().__init__()
        
        # Load BiomedCLIP model
        self.model, _ = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        
        # Extract encoders
        self.visual = self.model.visual  # TimmModel (ViT-B/16)
        self.text_encoder = self.model.text  # HFTextEncoder (BERT)
        
        # Access internal models
        self.vit = self.visual.trunk
        self.bert = self.text_encoder.transformer
        
        # Config
        self.hidden_size = 768
        self.patch_size = 16
        self.image_size = 224
        
        # Task-specific projection heads
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
        """Extract multi-scale vision features"""
        batch_size = x.shape[0]
        vit = self.vit
        
        # Patch embedding
        x = vit.patch_embed(x)  # [B, 196, 768]
        
        # Add CLS token + position embedding
        cls_token = vit.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = vit.pos_drop(x + vit.pos_embed)
        
        # Collect features from all blocks
        hidden_states = []
        for block in vit.blocks:
            x = block(x)
            hidden_states.append(x)
        
        x = vit.norm(x)
        cls_features = x[:, 0]  # [B, 768]
        
        # BiomedCLIP projection
        if hasattr(self.visual.head, 'proj'):
            clip_features = self.visual.head.proj(cls_features)
        else:
            clip_features = cls_features
        
        # Task-specific projection
        project_embed = self.vision_project_head(cls_features)
        
        # Convert to spatial features
        processed_features = []
        for hidden_state in hidden_states:
            patch_tokens = hidden_state[:, 1:, :]  # Remove CLS
            spatial = rearrange(patch_tokens, 'b (h w) c -> b c h w', h=14, w=14)
            processed_features.append(spatial)
        
        return {
            "feature": processed_features,      # [12 x [B, 768, 14, 14]]
            "project": project_embed,           # [B, project_dim]
            "clip_features": clip_features      # [B, 512]
        }
    
    def encode_text(self, input_ids, attention_mask=None):
        """Extract text features from BERT"""
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        
        # BERT embeddings
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=None
        )
        
        # Forward through BERT layers
        hidden_states = []
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        
        hidden = embedding_output
        for layer in self.bert.encoder.layer:
            hidden = layer(hidden, extended_attention_mask)[0]
            hidden_states.append(hidden)
        
        # Pool CLS token
        pooled_output = self.text_encoder.pooler(hidden, attention_mask)
        
        # BiomedCLIP projection
        if hasattr(self.text_encoder, 'proj'):
            clip_features = self.text_encoder.proj(pooled_output)
        else:
            clip_features = pooled_output
        
        # Task-specific projection
        project_embed = self.text_project_head(pooled_output)
        
        return {
            "feature": hidden_states,           # [12 x [B, L, 768]]
            "last_hidden_state": hidden,        # [B, L, 768]
            "pooled": pooled_output,            # [B, 768]
            "project": project_embed,           # [B, project_dim]
            "clip_features": clip_features      # [B, 512]
        }


class LanGuideMedSeg_BiomedCLIP(nn.Module):
    """
    Language-Guided Medical Segmentation using BiomedCLIP.
    Single shared encoder for memory efficiency.
    """
    
    def __init__(self, 
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 project_dim=512):
        super().__init__()
        
        # Shared BiomedCLIP encoder
        self.biomedclip = BiomedCLIPEncoder(model_name, project_dim)
        
        base_dim = 768
        
        # Multi-scale feature pyramid
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
        
        # Text-guided decoders
        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
    
    def forward(self, data):
        image, text = data
        
        # Convert grayscale to RGB if needed
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        
        # Encode image
        image_output = self.biomedclip.encode_image(image)
        vision_features = image_output['feature']
        
        # Encode text
        text_output = self.biomedclip.encode_text(
            text['input_ids'], 
            text.get('attention_mask')
        )
        text_guidance = text_output['last_hidden_state']
        
        # Multi-scale features (last 4 layers)
        selected_layers = [vision_features[8], vision_features[9],
                          vision_features[10], vision_features[11]]
        
        image_features = []
        for layer_name, layer_feature in zip(
            ['layer_9', 'layer_10', 'layer_11', 'layer_12'], selected_layers):
            
            transformed = self.feature_pyramid[layer_name](layer_feature)
            seq_feature = rearrange(transformed, 'b c h w -> b (h w) c')
            image_features.append(seq_feature)
        
        # Decoder with text guidance
        os32 = image_features[0]  # [B, 49, 768]
        os16 = self.decoder16(os32, image_features[1], text_guidance)
        os8 = self.decoder8(os16, image_features[2], text_guidance)
        os4 = self.decoder4(os8, image_features[3], text_guidance)
        
        # Upsample to full resolution
        os4 = rearrange(os4, 'B (H W) C -> B C H W', 
                       H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()
        
        return out


class LanGuideMedSeg_BiomedCLIP_WithContrastive(nn.Module):
    """
    Language-Guided Medical Segmentation with Contrastive Learning.
    """
    
    def __init__(self, 
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 project_dim=512):
        super().__init__()
        
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
        
        # Encode
        image_output = self.biomedclip.encode_image(image)
        vision_features = image_output['feature']
        image_clip = image_output['clip_features']
        
        if isinstance(text, dict):
            text_output = self.biomedclip.encode_text(
                text['input_ids'], 
                text.get('attention_mask')
            )
            text_guidance = text_output['last_hidden_state']  # [B, L, 768]
        else:
            # If text is already encoded tensor
            text_guidance = text
        
        # Multi-scale features
        selected_layers = [vision_features[8], vision_features[9],
                          vision_features[10], vision_features[11]]
        
        image_features = []
        for layer_name, layer_feature in zip(
            ['layer_9', 'layer_10', 'layer_11', 'layer_12'], selected_layers):
            
            transformed = self.feature_pyramid[layer_name](layer_feature)
            seq_feature = rearrange(transformed, 'b c h w -> b (h w) c')
            image_features.append(seq_feature)
        
        # Decode
        os32 = image_features[0]
        os16 = self.decoder16(os32, image_features[1], text_guidance)
        os8 = self.decoder8(os16, image_features[2], text_guidance)
        os4 = self.decoder4(os8, image_features[3], text_guidance)
        
        os4 = rearrange(os4, 'B (H W) C -> B C H W', 
                       H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()
        
        if return_embeddings:
            return out, image_clip, text_clip
        
        return out