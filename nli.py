from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class BertNLI:
    def __init__(
        self, model_name_or_path="Formzu/bert-base-japanese-jsnli", device=None
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.nli = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    @torch.no_grad()
    def predict(self, premise, hypothesis):
        input = self.tokenizer.encode(premise, hypothesis, return_tensors="pt").to(
            self.device
        )
        logits = self.nli(input)["logits"][0]
        probs = logits.softmax(dim=-1)
        probs = probs.cpu().numpy()
        label = self.label_map[np.argmax(probs, axis=0)]
        return label
