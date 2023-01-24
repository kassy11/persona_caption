import os
import random
import numpy as np
from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
import torch
from transformers import GPT2Config, GPT2DoubleHeadsModel, T5Tokenizer

ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}


class ConvAIModelJa(ConvAIModel):
    def __init__(self, model_name="rinna/japanese-gpt2-small", args=None, **kwargs):
        # NOTE: 親クラスのTokenizerクラス(GPT2Tokenizer)が
        # rinna/japanese-gpt2に対応していないため、親クラスのコンストラクタは呼び出さない

        # model_typeは決め打ちでgpt2とする
        model_type = "gpt2"
        MODEL_CLASSES = {
            "gpt2": (GPT2Config, GPT2DoubleHeadsModel, T5Tokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ConvAIArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 日本語版のモデルに合わせる
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.__dict__.update(kwargs)
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.tokenizer.do_lower_case = True
        self.config = config_class.from_pretrained(model_name, **self.args.config)

        if not self.args.quantized_model:
            self.model = model_class.from_pretrained(model_name)
        else:
            quantized_weights = torch.load(
                os.path.join(model_name, "pytorch_model.bin")
            )
            self.model = model_class.from_pretrained(
                None, config=self.config, state_dict=quantized_weights
            )

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

        self.add_special_tokens_(self.model, self.tokenizer)
        self.args.model_name = model_name
        self.args.model_type = model_type

        self.results = {}

    def add_special_tokens_(self, model, tokenizer):
        # tokenizerの語彙数
        orig_num_tokens = tokenizer.vocab_size
        # T5Tokenizerの特殊トークンをConvAIModelに合わせて上書きする
        num_added_tokens = tokenizer.add_special_tokens(
            ATTR_TO_SPECIAL_TOKEN
        )  # doesn't add if they are already there
        if num_added_tokens > 0:
            model.resize_token_embeddings(
                new_num_tokens=orig_num_tokens + num_added_tokens
            )
