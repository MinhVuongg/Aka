from enum import Enum

import torch


class ModelType(Enum):
    SEQ2SEQ = "seq2seq"
    CAUSAL = "causal"
    MASKED = "masked"
    ENCODER = "encoder"
    VISION = "vision"
    SPEECH = "speech"


def load_model_by_type(model_name, model_type: ModelType):
    if model_type == ModelType.SEQ2SEQ:
        from transformers import AutoModelForSeq2SeqLM
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
    elif model_type == ModelType.CAUSAL:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
    elif model_type == ModelType.MASKED:
        from transformers import AutoModelForMaskedLM
        return AutoModelForMaskedLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
    elif model_type == ModelType.ENCODER:
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
    elif model_type == ModelType.VISION:
        from transformers import AutoModelForImageClassification
        return AutoModelForImageClassification.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
    elif model_type == ModelType.SPEECH:
        from transformers import AutoModelForSpeechSeq2Seq
        return AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
