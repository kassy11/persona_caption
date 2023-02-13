import logging
from VLT5.vlt5_model import VLT5Model
from VLT5.vlt5_tokenizer import VLT5Tokenizer
import re
import torch
import spacy

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class Vqa:
    def __init__(
        self,
        normalize_boxes,
        roi_features,
        model_name_or_path="sonoisa/vl-t5-base-japanese",
        device=None,
    ):
        self.roi_features = roi_features
        self.normalized_boxes = normalize_boxes
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vlt5 = VLT5Model.from_pretrained(model_name_or_path)
        self.vlt5.to(self.device)

        self.tokenizer = VLT5Tokenizer.from_pretrained(
            model_name_or_path, max_length=24, do_lower_case=True
        )
        self.vlt5.resize_token_embeddings(self.tokenizer.vocab_size)
        self.vlt5.tokenizer = self.tokenizer

    def get_answer(self, questions):
        self.vlt5.eval()
        box_ids = set()
        answer_list = []

        nlp = spacy.load("ja_ginza")

        logger.info("Getting answers to my question about the image.")
        for question in questions:
            input_ids = self.tokenizer(
                question, return_tensors="pt", padding=True
            ).input_ids.to(self.device)
            vis_feats = self.roi_features.to(self.device)
            boxes = self.normalized_boxes.to(self.device)

            # Generate answers
            output = self.vlt5.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, boxes),
            )
            generated_sent = self.tokenizer.batch_decode(
                output, skip_special_tokens=False
            )[0]
            generated_sent = re.sub("[ ]*(<pad>|</s>)[ ]*", "", generated_sent)

            if "<vis_extra_id_" in generated_sent:
                match = re.match(r"<vis_extra_id_(\d+)>", generated_sent)
                box_id = int(match.group(1))
                box_ids.add(box_id)

            logger.info(f"{question}")
            logger.info(f"  -> {generated_sent}")

            doc = nlp(generated_sent)
            if ("何歳" or "年齢") in question:
                answer_list.append(self._get_age_answer(doc))
            else:
                for tok in doc:
                    if tok.pos_ in ("NOUN", "PRON", "PROPN", "ADJ", "VERB"):
                        answer_list.append(tok.text)
        return list(set(answer_list))

    def _get_age_answer(self, doc):
        for tok in doc:
            if tok.pos_ == "NUM":
                num = int(tok.text)
                if num >= 10 and num < 20:
                    return "10代"
                elif num >= 20 and num < 30:
                    return "20代"
                elif num >= 30 and num < 40:
                    return "30代"
                else:
                    return "老い"
            else:
                if ("幼" or "少" or "若") in tok.text:
                    return "10代"
                else:
                    return tok.text
