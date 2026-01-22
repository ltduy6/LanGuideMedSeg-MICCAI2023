import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)

        return {"feature":output['hidden_states']}
