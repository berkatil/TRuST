from model import TargetModel
from huggingface_hub import hf_hub_download
import torch

targets = ['no target', 'disability', 'black', 'native american', 'white',
            'refugee', 'other country', 'other', 'lgbtq+', 'woman',
            'united states', 'jewish', 'politics', 'asian', 'mexican',
            'chinese', 'other religion', 'man', 'muslim', 'middle east',
            'other ethnicity', 'latino', 'arab', 'other gender']

repo_id = "berkatil/TRuST_target_classifier"
weights_path = hf_hub_download(
    repo_id=repo_id,
    filename="target_classifier.pth"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = TargetModel(num_labels=len(targets), model="bert").to(device)
state_dict = torch.load(weights_path, map_location=device)
classifier.load_state_dict(state_dict)
classifier.eval()

example_test = "White people are good."
logits = classifier([example_test])
preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
preds = [targets[pred] for pred in preds]


