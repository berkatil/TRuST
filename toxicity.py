from model import TargetModel
from huggingface_hub import hf_hub_download
import torch


repo_id = "berkatil/TRuST_toxicity_classifier"
weights_path = hf_hub_download(
    repo_id=repo_id,
    filename="toxicity_classifier.pth"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = TargetModel(num_labels=2, model="bert").to(device)
state_dict = torch.load(weights_path, map_location=device)
classifier.load_state_dict(state_dict)
classifier.eval()

example_test = "White people are good."
logits = classifier([example_test])
preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
preds = ["toxic" if pred == 1 else "non-toxic" for pred in preds]



