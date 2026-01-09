import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionTokenMapper(nn.Module):
    def __init__(self, input_tokens=49, output_tokens=24, token_dim=768):
        super().__init__()
        
        # Learnable query tokens (24 medical concept queries)
        self.query_tokens = nn.Parameter(torch.randn(output_tokens, token_dim))
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Layer normalization and feed-forward
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim)
        )
        
    def forward(self, visual_tokens):
        """
        visual_tokens: [B, 49, 768]
        returns: [B, 24, 768]
        """
        batch_size = visual_tokens.size(0)
        
        # Expand query tokens for batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 24, 768]
        
        # Cross-attention: 24 queries attend to 49 visual tokens
        attended_tokens, _ = self.cross_attention(
            query=queries,
            key=visual_tokens,
            value=visual_tokens
        )
        
        # Residual connection and layer norm
        text_tokens = self.norm1(queries + attended_tokens)
        
        # Feed-forward network
        text_tokens = text_tokens + self.ffn(text_tokens)
        text_tokens = self.norm2(text_tokens)
        
        return text_tokens


class ImageToTextSemanticMapper(nn.Module):
    """
    Maps ConvNeXt Stage-4 image features to text-like semantic tokens.
    """
    def __init__(self, token_dim=768, num_text_tokens=24):
        super().__init__()

        # Global semantic extractor
        self.semantic_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim)
        )

        # Learnable expansion tokens (acts like text slots)
        self.text_slots = nn.Parameter(
            torch.randn(num_text_tokens, token_dim)
        )

        # Lightweight refinement attention
        self.refine_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=8,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, visual_tokens):
        """
        visual_tokens: [B, 49, 768]
        returns: [B, 24, 768]
        """

        B = visual_tokens.size(0)

        # ---- 1. Global semantic pooling ----
        image_proto = visual_tokens.mean(dim=1)      # [B, 768]
        image_proto = self.semantic_mlp(image_proto) # [B, 768]
        image_proto = F.normalize(image_proto, dim=-1)

        # ---- 2. Expand into text slots ----
        text_tokens = self.text_slots.unsqueeze(0).expand(B, -1, -1)
        text_tokens = text_tokens + image_proto.unsqueeze(1)

        # ---- 3. Optional refinement ----
        refined, _ = self.refine_attn(
            query=text_tokens,
            key=text_tokens,
            value=text_tokens
        )

        text_tokens = self.norm1(text_tokens + refined)
        text_tokens = self.norm2(text_tokens)

        return text_tokens
