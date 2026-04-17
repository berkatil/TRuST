from transformers import AutoModelForTokenClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch
from torch.utils.data import Dataset

class SpanDataset(Dataset):
    def __init__(self, texts, model_name="SpanBERT/spanbert-base-cased"):
        super().__init__()
        self.data = texts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _find_offsets(self, text, spans_raw):
        spans = spans_raw.split(",")
        spans = [span.strip() for span in spans]
        offsets = []
        text = text.lower()
        for span in spans:
            span = span.lower()
            start = text.find(span)
            if start == -1: 
                continue
            end = start + len(span)
            offsets.extend(list(range(start, end)))
        return offsets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=384
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets
            #"word_ids": enc.word_ids()
        }

repo_id = "berkatil/TRuST_span_classifier"
weights_path = hf_hub_download(
    repo_id=repo_id,
    filename="span_classifier.pth"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-base-cased", num_labels=2).to(device)
state_dict = torch.load(weights_path, map_location=device)
classifier.load_state_dict(state_dict)
classifier.eval()

example_test = "White people are bad."

dataset = SpanDataset([example_test])

for data in dataset:
    input_ids = data["input_ids"].to(classifier.device)
    attention_mask = data["attention_mask"].to(classifier.device)
    offset_mapping = data["offset_mapping"].squeeze(0).cpu().numpy()


    with torch.no_grad():
        outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    toxic_chars = []
    for i, (start, end) in enumerate(offset_mapping.tolist()):
        if attention_mask.squeeze(0)[i].item() == 0:
            continue  # skip padding
        if start == end:
            continue  # skip special tokens
        if preds[i] == 1:
            start, end = offset_mapping[i].tolist()
            toxic_chars.extend(list(range(start, end)))