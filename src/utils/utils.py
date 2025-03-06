import os
import torch
from src.config.config import ModelType

def normalize_path(path_str):
    normalized = path_str.replace('\\', '/')
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return os.path.normpath(normalized)

def load_model(model_name, model_type: ModelType):
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
