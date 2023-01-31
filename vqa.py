import logging
from VLT5.vlt5_model import VLT5Model
from VLT5.vlt5_tokenizer import VLT5Tokenizer
import re
import torch

logger = logging.getLogger(__name__)


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

            # TODO: 原形に直す&ストップワード削除
            answer_list.append(generated_sent)

        logger.info(
            "Successfully answer questions from the photo. answer list = %s",
            answer_list,
        )
        return answer_list
