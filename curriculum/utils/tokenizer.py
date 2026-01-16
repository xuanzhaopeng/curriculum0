
from typing import Optional
from transformers import AutoTokenizer, AutoProcessor, PreTrainedTokenizer, ProcessorMixin

def get_tokenizer(model_path: str, chat_template_path: Optional[str], **kwargs) -> PreTrainedTokenizer:
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if chat_template_path is not None:
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
    return tokenizer

def get_processor(model_path: str, chat_template_path: Optional[str], **kwards) -> ProcessorMixin:
    processor:ProcessorMixin = AutoProcessor.from_pretrained(model_path, **kwards)
    if chat_template_path is not None:
        with open(chat_template_path) as f:
            processor.chat_template = f.read()
    return processor