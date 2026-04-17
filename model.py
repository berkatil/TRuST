import torch
import transformers
from transformers import RobertaModel, BertModel, AutoTokenizer, AutoModel

class TargetModel(torch.nn.Module):
    def __init__(self, num_labels, model="roberta"):
        super(TargetModel, self).__init__()
        self.model_name = model
        if model == "roberta":
            self.embedder = RobertaModel.from_pretrained('roberta-base')
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
        elif model == "bert":
            self.embedder = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        elif model == "modern":
            self.embedder = AutoModel.from_pretrained('answerdotai/ModernBERT-base')
            self.tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
            

        self.dense = torch.nn.Linear(self.embedder.config.hidden_size, num_labels)
        

    def forward(self, input_text):
        tokens = self.tokenizer(input_text, return_tensors='pt', padding="max_length", truncation=True).to(self.embedder.device)
        outputs = self.embedder(**tokens)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        logits = self.dense(cls_output)
        return logits