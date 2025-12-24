import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


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
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512, dataset_size=None):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

        self.dataset_size = dataset_size
        if dataset_size is not None:
            self.register_buffer('query_space', torch.zeros(dataset_size, project_dim))
            self.register_buffer('response_space', torch.zeros(dataset_size, 24, 768))
            self.register_buffer('update_mask', torch.zeros(dataset_size, dtype=torch.bool))
        else:
            self.query_space = None
            self.response_space = None
            self.update_mask = None
    
    def update_spaces(self, image_project, text_embeds, idx):
        if (self.query_space is None) or (self.response_space is None):
            return

        idx = idx.long().to(self.query_space.device)

        with torch.no_grad():
            self.query_space[idx] = image_project.detach()
            self.response_space[idx] = text_embeds.detach()
            self.update_mask[idx] = True

    def retrieve(self, image_project, top_k=1):
        """
        Retrieve text embeddings based on image project similarity during validation/testing
        
        Args:
            image_project (torch.Tensor): Image projection features [B, project_dim] or [project_dim]
            top_k (int): Number of top similar entries to retrieve (default=1)
            
        Returns:
            torch.Tensor: Retrieved text embeddings [B, top_k, 24, 768] or [B, 24, 768] if top_k=1
            torch.Tensor: Similarity scores [B, top_k] or [B] if top_k=1
            torch.Tensor: Retrieved indices [B, top_k] or [B] if top_k=1
        """

        if self.query_space is None or self.response_space is None:
            print("Warning: query_space or response_space is not initialized.")
            return None, None, None
        
        if not self.update_mask.any():
            print("Warning: No entries in the memory have been updated yet.")
            return None, None, None

        if len(image_project.shape) == 1:
            image_project = image_project.unsqueeze(0)  # Make it [1, project_dim]

        batch_size = image_project.shape[0]

        # Only consider updated entries in the memory
        valid_indices = torch.where(self.update_mask)[0]
        valid_query_space = self.query_space[valid_indices]
        valid_response_space = self.response_space[valid_indices]

        if len(valid_indices) == 0:
            print("Warning: No valid entries in the memory to retrieve from.")
            return None, None, None
        
        # Normalize for cosine similarity
        image_project_norm = F.normalize(image_project, p=2, dim=1)
        valid_query_norm = F.normalize(valid_query_space, p=2, dim=1)

        # Compute cosine similarity
        similarity_scores = torch.mm(image_project_norm, valid_query_norm.t())  # [B, N_valid]

        # Get top-k most similar entries
        top_k = min(top_k, len(valid_indices))
        top_scores, top_indices_in_valid = torch.topk(similarity_scores, k=top_k, dim=1)

        # Map back to original indices
        top_original_indices = valid_indices[top_indices_in_valid]

        # Retrieve corresponding text embeddings
        retrieved_text_embeds = []
        for i in range(batch_size):
            batch_retrieved = valid_response_space[top_indices_in_valid[i]]
            retrieved_text_embeds.append(batch_retrieved)
        
        retrieved_text_embeds = torch.stack(retrieved_text_embeds)  # [B, top_k, 24, 768]

        if top_k == 1:
            retrieved_text_embeds = retrieved_text_embeds.squeeze(1)  # [B, 24, 768]
            
        return retrieved_text_embeds
    
    def forward(self, data):

        image, text, idx = data
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
        
        if self.training:
            self.update_spaces(image_project, text_embeds[-1], idx)
            text_guidance = text_embeds[-1]
        else:
            retrieved_text_embeds = self.retrieve(image_project, top_k=1)
            if retrieved_text_embeds is not None:
                text_guidance = retrieved_text_embeds
            else:
                text_guidance = text_embeds[-1]

        print(text_guidance.shape)
        os16 = self.decoder16(os32,image_features[2], text_guidance)
        os8 = self.decoder8(os16,image_features[1], text_guidance)
        os4 = self.decoder4(os8,image_features[0], text_guidance)
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out
    
