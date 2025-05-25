from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        return F.normalize(outputs.pooler_output, dim=-1)