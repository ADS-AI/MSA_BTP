from .cfg_reader import CfgNode
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from ...datasets import text
# separate classification and regression

def load_untrained_model(cfg , model_name):
    print("Loading new theif model ", model_name)
    num_labels = cfg.NUM_LABELS
    if model_name.lower() == "bert-base-uncased":
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased" , num_labels = num_labels)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = AutoConfig.from_pretrained("bert-base-uncased")
    elif model_name.lower() == "roberta-base":
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base" , num_labels = num_labels)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        config = AutoConfig.from_pretrained("roberta-base")
    elif model_name.lower() == "distilbert":
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased" , num_labels = num_labels)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        config = AutoConfig.from_pretrained("distilbert-base-uncased")
    elif model_name.lower() == "xlnet":
        model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased" , num_labels = num_labels)
        tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
        config = AutoConfig.from_pretrained("xlnet-base-cased")
    elif model_name.lower() == "albert":
        model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2" , num_labels = num_labels)
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        config = AutoConfig.from_pretrained("albert-base-v2")
    elif model_name.lower() == "gpt2":
        model = AutoModelForSequenceClassification.from_pretrained("gpt2" , num_labels = num_labels)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = AutoConfig.from_pretrained("gpt2")

    return model , tokenizer , config


def load_victim_model(cfg: CfgNode):
    MODEL_DIR = cfg.VICTIM_MODEL_DIR
    config = AutoConfig.from_pretrained(
            MODEL_DIR,
            num_labels=cfg.VICTIM.NUM_LABELS, 
            finetuning_task=cfg.VICTIM.TASK_NAME, 
            cache_dir=cfg.CACHE_DIR if cfg.CACHE_DIR else None)
    tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, 
            do_lower_case=cfg.VICTIM.DO_LOWER_CASE, 
            cache_dir=cfg.CACHE_DIR if cfg.CACHE_DIR else None)
    model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, 
            from_tf=bool(".ckpt" in MODEL_DIR), 
            config=config, 
            cache_dir=cfg.CACHE_DIR if cfg.CACHE_DIR else None)
    return model , tokenizer , config

def load_thief_model(cfg: CfgNode):
    MODEL_DIR = cfg.THIEF_MODEL_DIR
    config = AutoConfig.from_pretrained(
            MODEL_DIR,
            num_labels=cfg.VICTIM.NUM_LABELS, 
            finetuning_task=cfg.VICTIM.TASK_NAME, 
            cache_dir=cfg.CACHE_DIR if cfg.CACHE_DIR else None)
    tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, 
            do_lower_case=cfg.VICTIM.DO_LOWER_CASE, 
            cache_dir=cfg.CACHE_DIR if cfg.CACHE_DIR else None)
    model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, 
            from_tf=bool(".ckpt" in MODEL_DIR), 
            config=config, 
            cache_dir=cfg.CACHE_DIR if cfg.CACHE_DIR else None)
    return model , tokenizer , config

def load_dataset_thief(cfg):
    