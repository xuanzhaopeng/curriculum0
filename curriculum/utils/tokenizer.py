
from transformers import AutoTokenizer, AutoProcessor, PreTrainedTokenizer, ProcessorMixin

def get_tokenizer(model_path: str, **kwargs) -> PreTrainedTokenizer:
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    return tokenizer

def get_processor(model_path: str, **kwards) -> ProcessorMixin:
    processor:ProcessorMixin = AutoProcessor.from_pretrained(model_path, **kwards)
    return processor