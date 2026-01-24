import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import open_clip

class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)

        return {'feature':output.last_hidden_state}

class BiomedCLIPBERTModel(nn.Module):
    def __init__(self):
        super(BiomedCLIPBERTModel, self).__init__()

        self.biomedclip = open_clip.create_model('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.biomedclip.requires_grad_(False)
    
    def forward(self, input_ids, attention_mask):
        bert = self.biomedclip.text

        out = bert.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return {'feature':out.last_hidden_state}