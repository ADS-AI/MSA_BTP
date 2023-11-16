import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from MSA_BTP.msa_toolbox.utils.text.load_data_and_models import load_untrained_model, load_dataset_thief
from . cfg_reader import CfgNode
# import samplers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from ...models.text.model import BertForSequenceClassificationDistil

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "dstilbert": (BertConfig, BertForSequenceClassificationDistil, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

def set_seed(cfg , n_gpu):
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.SEED)
        
def simple_accuracy(preds, labels):
        return (preds == labels).mean()

def active_learning_technique(victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config, cfg):
    # if cfg.ACTIVE.METHOD == "qbc_stealing":
    #     return qbc_stealing(cfg,None, thief_model)
    if cfg.ACTIVE.METHOD == "entroby_stealing":
        return entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config)
    # elif cfg.ACTIVE.METHOD == "all_data_stealing":
    #     return all_data_stealing(cfg, thief_model, unlabeled_loader)

def query_data_in_victim(cfg , theif_dataloader, victim_model, victim_tokenizer, victim_config):
    '''
    This function is used to query all the data from the victim model
    returns- 
    index_to_new_label - dictionary mapping index to new label
    index_to_entropy -  dictionary mapping index to entropy
    orignal_index - list of original index
    '''
    print("--------------- victim model setup for querying ---------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    victim_model.to(device)
    
    print("--------------- querying ---------------")
    new_labels = None
    orignal_labels = None
    orignal_index = None
    entropy = None
    for batch in tqdm(theif_dataloader, desc="Evaluating"):
        victim_model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
            outputs = victim_model(**inputs)
            logits = outputs['logits']
        if new_labels is None:
            new_labels = logits.detach().cpu().numpy()
            orignal_labels = batch[4].detach().cpu().numpy()
            orignal_index = batch[0].detach().cpu().numpy()
            entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            entropy = np.sum(entropy * np.log(entropy), axis=1)
        else:
            new_labels = np.append(new_labels, logits.detach().cpu().numpy(), axis=0)
            orignal_labels = np.append(orignal_labels, batch[4].detach().cpu().numpy(), axis=0)
            orignal_index = np.append(orignal_index, batch[0].detach().cpu().numpy(), axis=0)
            tmp_entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            tmp_entropy = np.sum(tmp_entropy * np.log(tmp_entropy), axis=1)
            entropy = np.append(entropy, tmp_entropy, axis=0)

    new_labels = np.argmax(new_labels, axis=1)
    index_to_new_label = {}
    index_to_entropy = {}

    for i in range(len(orignal_index)):
        index_to_new_label[orignal_index[i]] = new_labels[i]
        index_to_entropy[orignal_index[i]] = entropy[i]

    print("--------------- querying done ---------------")
    return index_to_new_label, index_to_entropy , orignal_index
   
def get_entropy(cfg , model , tokenizer , config , n_gpu , device , dataloader):
    entropy = None
    orignal_index = None
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
            outputs = model(**inputs)
            logits = outputs['logits']
            # print("logits", logits)
        if entropy is None:
            orignal_index = batch[0].detach().cpu().numpy()
            entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            entropy = np.sum(entropy * np.log(entropy), axis=1)
        else:
            orignal_index = np.append(orignal_index, batch[0].detach().cpu().numpy(), axis=0)
            tmp_entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            tmp_entropy = np.sum(tmp_entropy * np.log(tmp_entropy), axis=1)
            entropy = np.append(entropy, tmp_entropy, axis=0)
    
    index_to_entropy = {}
    for i in range(len(orignal_index)):
        index_to_entropy[orignal_index[i]] = entropy[i]
    
    return index_to_entropy

def active_train(cfg , train_dataloader , eval_dataloader , device , n_gpu , index_to_new_label , cycle_num  , model , config , tokenizer,  prefix = ""):
    if cfg.TRAIN.MAX_STEPS > 0:
        t_total = cfg.TRAIN.MAX_STEPS
        cfg.TRAIN.EPOCH = cfg.TRAIN.MAX_STEPS // (len(train_dataloader) // cfg.TRAIN.GRAD_ACCUM_STEPS) + 1
    else:
        t_total = len(train_dataloader) // cfg.TRAIN.GRAD_ACCUM_STEPS * cfg.TRAIN.EPOCH
    
    # config, model, tokenizer = get_model(cfg , type="thief" , prefix = prefix)
    model.to(device)
    bert_embed = model.get_input_embeddings()
    vocab_size, embed_size = bert_embed.weight.size()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.TRAIN.WDECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    if(cfg.TRAIN.OPTIMIZER == "Adam"):
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.TRAIN.LR, eps=cfg.TRAIN.ADAM_EPS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.TRAIN.WARMUP_STEPS, num_training_steps=t_total
    )

    print("***** Running training *****")
    print("  Num examples = ", len(train_dataloader))
    print("  Num Epochs = ", cfg.TRAIN.EPOCH)
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(cfg.TRAIN.EPOCH), desc="Epoch", disable=cfg.LOCAL_RANK not in [-1, 0],
    )
    set_seed(cfg, n_gpu)  # Added here for reproductibility (even between python 2 and 3)  
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.LOCAL_RANK not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            new_labels = []
            for i in range(len(batch[0])):
                new_labels.append(index_to_new_label[batch[0][i].item()])
            new_labels = torch.tensor(new_labels).to(device)
            inputs = {"input_ids": batch[1], "attention_mask": batch[2], "labels": new_labels}
            if cfg.VICTIM.MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = (
                        batch[3] if cfg.VICTIM.MODEL_TYPE in ["bert", "dpbert", "xlnet", "albert", "dstilbert"] else None
                )  
            outputs = model(**inputs)
            loss = outputs[0]  
            if n_gpu > 1:
                loss = loss.mean()
            if cfg.TRAIN.GRAD_ACCUM_STEPS > 1:
                loss = loss / cfg.TRAIN.GRAD_ACCUM_STEPS

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) %  cfg.TRAIN.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
                global_step += 1

    
    epoch_iterator.close()
    train_iterator.close()
    # Save model checkpoint
    output_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix)  , "cycle_num-{}".format(cycle_num) )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(cfg, os.path.join(output_dir, "training_cfg.bin"))
    print("Saving model checkpoint to ", output_dir)
            
    acc = evaluate(cfg, eval_dataloader, model, tokenizer, config, n_gpu, device , index_to_new_label = index_to_new_label)
    with open(os.path.join(output_dir , "cycle_num-{}.txt".format(cycle_num)), "w") as writer :
        writer.write("Eval Accuracy - {}\n".format(acc))
        writer.write("Train_examples - {}\n".format(len(train_dataloader)))
        writer.write("Train_examples - {}\n".format(len(eval_dataloader)))
        
    eval_results = {}
    eval_results["acc"] = acc
    return global_step, tr_loss / global_step , eval_results , model , config , tokenizer 

def evaluate(cfg, DataLoader, model, tokenizer, config, n_gpu=1 , device=None , index_to_new_label = None ):
    print("***** Running evaluation *****")
    results = {}
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    if index_to_new_label is None:
        for batch in tqdm(DataLoader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[1], "attention_mask": batch[2], "labels": batch[4]}
                if cfg.THIEF.MODEL_TYPE != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3] if cfg.THIEF.MODEL_TYPE in ["bert", "dpbert", "xlnet", "albert", "dstilbert"] else None)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy() , axis=0)
        nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        results = {"acc": simple_accuracy(preds, out_label_ids)}
    else:
        for batch in tqdm(DataLoader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            new_labels = []
            for i in range(len(batch[0])):
                new_labels.append(index_to_new_label[batch[0][i].item()])
            new_labels = torch.tensor(new_labels).to(device)

            with torch.no_grad():
                inputs = {"input_ids": batch[1], "attention_mask": batch[2], "labels": new_labels}
                if cfg.VICTIM.MODEL_TYPE != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3] if cfg.VICTIM.MODEL_TYPE in ["bert", "dpbert", "xlnet", "albert", "dstilbert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = {"acc": simple_accuracy(preds, out_label_ids)}
        results.update(result)
    return results

def entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format("hello") , cfg.THIEF.MODEL_TYPE)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    validation_accuracy_dict = {}
    global_step_dict = {}
    loss_dict = {}
    thief_model.to(device)  
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    thief_model.save_pretrained(model_out_dir)
    thief_tokenizer.save_pretrained(model_out_dir)
    thief_config.save_pretrained(model_out_dir)

    thief_dataset_train_features = thief_dataset.get_features(split = 'train', tokenizer = thief_tokenizer)
    BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
    sampler = RandomSampler(thief_dataset_train_features)
    train_dataloader = DataLoader(thief_dataset_train_features, sampler=sampler, batch_size=BATCH_SIZE)
    
    if cfg.THIEF.DO_EVALUATION == True:
        thief_dataset_val_features = thief_dataset.get_features(split = 'val', tokenizer = thief_tokenizer)
        validation_sampler = SequentialSampler(thief_dataset_val_features)
        validation_dataloader = DataLoader(thief_dataset_val_features, sampler=validation_sampler, batch_size=BATCH_SIZE)
    
    if cfg.THIEF.DO_TRAINING == True:
        print("---------------  quering victim thief_model using thief data & created indexes ---------------")
        Thief_dataset_train , index_to_new_label_train, index_to_entropy_train , index_list_train =  query_data_in_victim(cfg=cfg, theif_dataloader=train_dataloader, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)
        if cfg.THIEF.DO_EVALUATION == True:
            Thief_dataset_val , index_to_new_label_val, index_to_entropy_val , index_list_val =  query_data_in_victim(cfg=cfg, theif_dataloader=validation_dataloader, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)

        budget = cfg.ACTIVE.BUDGET
        unlabeled_index_list = list(index_list_train)
        labelled_index_list = []
        validation_index_list = []
        if True:
            unlabeled_index_list = random.sample(unlabeled_index_list, len(unlabeled_index_list))
            labelled_index_list.extend(unlabeled_index_list[0:1000])
            validation_index_list.extend(unlabeled_index_list[1000:2000])
            unlabelled_index_set = set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list)
            unlabelled_index_list = list(unlabelled_index_set)

        budget = budget - 2000
        cycle_num = 1
        while(budget >= 0):
            print("--------------- cycle {} budget spent -{}---------------".format(cycle_num , budget))
            thief_dataset_train_features_select = thief_dataset.get_features( split = 'train', tokenizer = thief_tokenizer, indexes = labelled_index_list)
            thief_dataset_val_features_select = thief_dataset.get_features( split = 'val', tokenizer = thief_tokenizer, indexes = validation_index_list)
            BATCH_SIZE_TRAIN = cfg.VICTIM.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
            BATCH_SIZE_EVAL = cfg.VICTIM.PER_GPU_EVAL_BATCH_SIZE * max(1, n_gpu)
            train_sampler = RandomSampler(thief_dataset_train_features_select)
            train_loader = DataLoader(thief_dataset_train_features_select, sampler=train_sampler, batch_size=BATCH_SIZE_TRAIN)
            validation_sampler = SequentialSampler(thief_dataset_val_features_select)
            validation_loader = DataLoader(thief_dataset_val_features_select, sampler=validation_sampler, batch_size=BATCH_SIZE_EVAL)
            global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , train_loader , validation_loader , device , n_gpu , index_to_new_label_train, cycle_num , thief_model , thief_config , thief_tokenizer,  "hello")
            validation_accuracy_dict[cycle_num] = validation_accuracy
            global_step_dict[cycle_num] = global_step
            loss_dict[cycle_num] = tr_loss
            print("global_step = %s, average_training_loss = %s" % (global_step, tr_loss))
            print("validation accuracy", validation_accuracy)
            thief_model.save_pretrained(model_out_dir)
            thief_tokenizer.save_pretrained(model_out_dir)
            thief_config.save_pretrained(model_out_dir)
            
            # select the top 1000 data
            if budget >= 1000:
                unlabelled_dataset = thief_dataset.get_features(split = 'train', tokenizer = thief_tokenizer, indexes = unlabelled_index_list)
                unlabelled_sampler = SequentialSampler(unlabelled_dataset)
                unlabelled_loader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                print("--------------- get the entropy of the unlabelled data ---------------")
                index_to_entropy = get_entropy(cfg , thief_model , thief_tokenizer , thief_config , n_gpu , device , unlabelled_loader)
                unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
                labelled_index_list.extend(unlabelled_index_list[:1000])
                unlabelled_index_list = unlabelled_index_list[1000:]
                budget = budget - 1000
            elif budget == 0:
                break
            elif budget >= len(unlabelled_index_list) & budget < 1000:
                unlabelled_dataset = thief_dataset.get_features(split = 'train', tokenizer = thief_tokenizer, indexes = unlabelled_index_list)
                unlabelled_sampler = SequentialSampler(unlabelled_dataset)
                unlabelled_loader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                print("--------------- get the entropy of the unlabelled data ---------------")
                index_to_entropy = get_entropy(cfg , thief_model , thief_tokenizer , thief_config , n_gpu , device , unlabelled_loader)
                unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
                budget = len(unlabelled_index_list)
                labelled_index_list.extend(unlabelled_index_list[:budget])  
                unlabelled_index_list = unlabelled_index_list[budget:]
                budget = -1
            else:
                unlabelled_dataset = thief_dataset.get_features(split = 'train', tokenizer = thief_tokenizer, indexes = unlabelled_index_list)
                unlabelled_sampler = SequentialSampler(unlabelled_dataset)
                unlabelled_loader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                print("--------------- get the entropy of the unlabelled data ---------------")
                index_to_entropy = get_entropy(cfg , thief_model , thief_tokenizer , thief_config , n_gpu , device , unlabelled_loader)
                unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
                labelled_index_list.extend(unlabelled_index_list[:budget])
                unlabelled_index_list = unlabelled_index_list[budget:]
                budget = -1
            print("------------------- budget left -",budget, "-------------------")
            cycle_num = cycle_num + 1
    

    with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
        f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\t")
        for cycle_num in validation_accuracy_dict.keys():
            f.write("%s\t%s\t%s\t%s\t" % (cycle_num, global_step_dict[cycle_num], loss_dict[cycle_num], validation_accuracy_dict[cycle_num]))
    print(validation_accuracy_dict)

    if cfg.THIEF.DO_TESTING == True:
        Dataset_list = Dataset.__dict__.keys()
        if cfg.VICTIM.DATASET not in Dataset_list:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(Dataset_list))
        Dataset_victim = Dataset.__dict__[cfg.VICTIM.DATASET]
        Dataset_victim = Dataset_victim(cfg.VICTIM.DATASET , cfg.VICTIM.DATA_DIR, cfg.VICTIM.NUM_LABELS , label_probs = False, Mode = cfg.VICTIM.DATA_MODE , args = cfg)
        if cfg.VICTIM.DATASET != 'sst2':
            victim_examples = Dataset_victim.get_test_examples()
        else:
            victim_examples = Dataset_victim.get_dev_examples()
        victim_label_list = Dataset_victim.get_labels()
        victim_data_test = victim_dataset.get_features(split = "test", thief_tokenizer = thief_tokenizer)
        sampler = RandomSampler(victim_data_test)
        BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
        victim_dataloader = DataLoader(victim_data_test, sampler=sampler, batch_size=BATCH_SIZE)

        accuracy_list = {}
        model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format("hello") )
        model_folders = os.listdir(model_out_dir)
        model_folders.sort()
        model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
        print(model_folders)
        for model_folder in model_folders:
            print("Evaluating thief_model:", model_folder)
            thief_model = AutoModelForSequenceClassification.from_pretrained(model_folder)
            thief_tokenizer = AutoTokenizer.from_pretrained(model_folder)
            thief_config = AutoConfig.from_pretrained(model_folder)
            thief_model.to(device)
            result = evaluate(cfg, victim_dataloader, thief_model, thief_tokenizer, thief_config, n_gpu , device)
            accuracy_list[model_folder] = result 
        print("accuracy list", accuracy_list)
        with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
            for key, value in accuracy_list.items():
                f.write("%s,%s" % (key, value))
                f.write("\n")
        return accuracy_list

# def All_Data_Stealing(cfg , n_gpu , device , prefix = ""):
#     config = AutoConfig.from_pretrained( cfg.THIEF.MODEL_NAME_OR_PATH,num_labels= cfg.VICTIM.NUM_LABELS ,finetuning_task=cfg.THIEF.TASK_NAME)
#     tokenizer = AutoTokenizer.from_pretrained(cfg.THIEF.MODEL_NAME_OR_PATH,do_lower_case=cfg.THIEF.DO_LOWER_CASE,cache_dir=cfg.THIEF.CACHE_DIR if cfg.THIEF.CACHE_DIR else None)
#     model = AutoModelForSequenceClassification.from_pretrained(cfg.THIEF.MODEL_NAME_OR_PATH, from_tf=bool(".ckpt" in cfg.THIEF.MODEL_NAME_OR_PATH),config=config)
#     model.to(device)

#     thief_datasets = Dataset.__dict__.keys()
#     if cfg.THIEF.DATASET not in thief_datasets:
#         raise ValueError('Dataset not found. Valid arguments = {}'.format(thief_datasets))
#     Thief_Dataset = Dataset.__dict__[cfg.THIEF.DATASET]
#     Thief_Dataset = Thief_Dataset(cfg.THIEF.DATASET , cfg.THIEF.DATA_DIR, cfg.THIEF.NUM_LABELS , label_probs = False, Mode = cfg.THIEF.DATA_MODE , args = cfg)
#     train_examples = Thief_Dataset.get_train_examples()
#     train_label_list = Thief_Dataset.get_labels()
#     train_features = convert_examples_to_features(train_examples,tokenizer,label_list=train_label_list,
#         max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
#         pad_on_left=bool(cfg.VICTIM.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
#         pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
#         pad_token_segment_id=4 if cfg.VICTIM.MODEL_TYPE in ["xlnet"] else 0)
#     train_data = load_and_cache_examples(cfg, train_features)
#     sampler = RandomSampler(train_data)
#     BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
#     train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=BATCH_SIZE)
    
#     if cfg.THIEF.DO_EVALUATION == True:
#         val_examples = Thief_Dataset.get_dev_examples()
#         val_label_list = Thief_Dataset.get_labels()
#         val_features = convert_examples_to_features(val_examples,tokenizer,label_list=val_label_list,
#                 max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
#                 pad_on_left=bool(cfg.THIEF.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
#                 pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
#                 pad_token_segment_id=4 if cfg.THIEF.MODEL_TYPE in ["xlnet"] else 0)
#         val_data = load_and_cache_examples(cfg, val_features)
#         sampler = RandomSampler(val_data)
#         BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
#         val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=BATCH_SIZE)
    

#     if cfg.THIEF.DO_TRAINING == True:
#         print("---------------  quering victim model using thief data & created indexes ---------------")
#         model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix)  , cfg.THIEF.MODEL_TYPE)
#         if not os.path.exists(model_out_dir):
#             os.makedirs(model_out_dir)
#         index_to_new_label_train, index_to_entropy_train , index_list_train =  query_all_data_thief(cfg=cfg, thief_dataloader=train_dataloader, n_gpu=n_gpu, device=device)
#         if cfg.THIEF.DO_EVALUATION == True:
#             index_to_new_label_val, index_to_entropy_val , index_list_val =  query_all_data_thief(cfg=cfg, thief_dataloader=val_dataloader, n_gpu=n_gpu, device=device)
#         print("--------------- Train thief with all data ---------------")
#         # prefix = ""
#         if cfg.THIEF.DO_EVALUATION == True:
#             global_step, tr_loss , model , tokenizer , config = train_thief(cfg,  train_dataloader, val_dataloader,  model, tokenizer, config , n_gpu , device , index_to_new_label=index_to_new_label_train, index_to_new_label_val = index_to_new_label_val , prefix = prefix)
#         else:
#             print("--------------- model setup ---------------") 
#             global_step, tr_loss , model , tokenizer , config = train_thief(cfg,  train_dataloader, None,  model, tokenizer, config , n_gpu , device , index_to_new_label=index_to_new_label_train, prefix = prefix)
        
#         print("Saving model to", model_out_dir)
#         model_to_save = (model.module if hasattr(model, "module") else model)
#         model_to_save.save_pretrained(model_out_dir)
#         tokenizer.save_pretrained(model_out_dir)
#         torch.save(cfg, os.path.join(model_out_dir, "training_args.bin"))
#         print("global_step = %s, average loss = %s" % (global_step, tr_loss))
   
#     if cfg.THIEF.DO_TESTING == True:
#         Dataset_list = Dataset.__dict__.keys()
#         if cfg.VICTIM.DATASET not in Dataset_list:
#             raise ValueError('Dataset not found. Valid arguments = {}'.format(Dataset_list))
#         Dataset_victim = Dataset.__dict__[cfg.VICTIM.DATASET]
#         Dataset_victim = Dataset_victim(cfg.VICTIM.DATASET , cfg.VICTIM.DATA_DIR, cfg.VICTIM.NUM_LABELS , label_probs = False, Mode = cfg.VICTIM.DATA_MODE , args = cfg)
#         if cfg.VICTIM.DATASET != 'sst2':
#             victim_examples = Dataset_victim.get_test_examples()
#         else:
#             victim_examples = Dataset_victim.get_dev_examples()
#         victim_label_list = Dataset_victim.get_labels()
#         victim_features = convert_examples_to_features(victim_examples,tokenizer,label_list=victim_label_list,
#             max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
#             pad_on_left=bool(cfg.THIEF.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
#             pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
#             pad_token_segment_id=4 if cfg.THIEF.MODEL_TYPE in ["xlnet"] else 0)
#         victim_data = load_and_cache_examples(cfg, victim_features)
#         sampler = RandomSampler(victim_data)
#         BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
#         victim_dataloader = DataLoader(victim_data, sampler=sampler, batch_size=BATCH_SIZE)

#         accuracy_list = {}
#         model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix))
#         model_folders = os.listdir(model_out_dir)
#         model_folders.sort()
#         model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
#         print(model_folders)
#         for model_folder in model_folders:
#             print("Evaluating model:", model_folder)
#             model = AutoModelForSequenceClassification.from_pretrained(model_folder)
#             tokenizer = AutoTokenizer.from_pretrained(model_folder)
#             config = AutoConfig.from_pretrained(model_folder)
#             model.to(device)
#             result = evaluate(cfg, victim_dataloader, model, tokenizer, config, n_gpu , device)
#             accuracy_list[model_folder] = result 
#         print("accuracy list", accuracy_list)
#         with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
#             for key, value in accuracy_list.items():
#                 f.write("%s,%s" % (key, value))
#                 f.write("\n")
  
# def qbc_stealing(cfg : CfgNode , device: torch.device() , thief_model: nn.Module , list_thief_models : list):
    
#     validation_accuracy= [{}]*cfg.ACTIVE.NUM_MODELS
#     global_steps = [{}]*cfg.ACTIVE.NUM_MODELS
#     losses = [{}]*cfg.ACTIVE.NUM_MODELS

#     thief_models = {}
#     thief_config = {}
#     thief_tokenizer = {}

    

#     if cfg.THIEF.DO_TRAINING == True:
#         for i in range(cfg.ACTIVE.NUM_MODELS):
#             thief_model[i] , thief_config[i] , thief_tokenizer[i] = load_untrained_model(cfg, list_thief_models[i])


#     if cfg.THIEF.DO_TRAINING == True:
#         print("---------------  quering victim model using thief data & created indexes ---------------")
#         Thief_dataset_train , index_to_new_label_train, index_to_entropy_train , index_list_train =  query_all_data_thief(cfg=cfg, n_gpu=n_gpu, device=device , split='train')
#         if cfg.THIEF.DO_EVALUATION == True:
#             Thief_dataset_val , index_to_new_label_val, index_to_entropy_val , index_list_val =  query_all_data_thief(cfg=cfg, n_gpu=n_gpu, device=device , split='val') 
#         budget = cfg.ACTIVE.BUDGET
#         unlabeled_index_list = list(index_list_train)
#         labelled_index_list = []
#         validation_index_list = []
#         if True:
#             unlabeled_index_list = random.sample(unlabeled_index_list, len(unlabeled_index_list))
#             labelled_index_list.extend(unlabeled_index_list[0:1000])
#             validation_index_list.extend(unlabeled_index_list[1000:2000])
#             unlabelled_index_set = set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list)
#             unlabelled_index_list = list(unlabelled_index_set)
#         budget = budget - 2000
#         cycle_num = 1
#         examples = Thief_dataset_train.get_train_examples()
#         label_list = Thief_dataset_train.get_labels()
#         print("-----Creating features from dataset-----")
#         features = convert_examples_to_features(examples,thief_tokenizer,label_list=label_list,
#             max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
#             pad_on_left=bool(cfg.VICTIM.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
#             pad_token=thief_tokenizer.convert_tokens_to_ids([thief_tokenizer.pad_token])[0],
#             pad_token_segment_id=4 if cfg.VICTIM.MODEL_TYPE in ["xlnet"] else 0)
#         print("-----features created")
#         index_to_entropy = {}
#         index_to_label_prob = {}

#         while(budget >= 0):
#             for i in range(num_models):
#                 model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(i)  , cfg.THIEF.MODEL_TYPE)
#                 print("--------------- cycle {} budget spent -{}---------------".format(cycle_num , budget))
#                 training_dataset = load_and_cache_examples_by_index(cfg , labelled_index_list , features)
#                 validation_dataset = load_and_cache_examples_by_index(cfg , validation_index_list , features)
#                 BATCH_SIZE_TRAIN = cfg.VICTIM.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
#                 BATCH_SIZE_EVAL = cfg.VICTIM.PER_GPU_EVAL_BATCH_SIZE * max(1, n_gpu)
#                 train_sampler = SequentialSampler(training_dataset)
#                 train_dataloader = DataLoader(training_dataset, sampler=train_sampler, batch_size=BATCH_SIZE_TRAIN)
#                 validation_sampler = SequentialSampler(validation_dataset)
#                 validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=BATCH_SIZE_EVAL)
#                 global_step, tr_loss , validation_accuracy , model , config , tokenizer = active_train(cfg , train_dataloader , validation_dataloader , device , n_gpu , index_to_new_label_train, cycle_num , i)
#                 validation_accuracy_dict_all_models[i][cycle_num] = validation_accuracy
#                 global_step_dict_all_models[i][cycle_num] = global_step
#                 loss_dict_all_models[i][cycle_num] = tr_loss
#                 print("global_step = %s, average_training_loss = %s" % (global_step, tr_loss))
#                 print("validation accuracy", validation_accuracy)
#                 model.save_pretrained(model_out_dir)
#                 tokenizer.save_pretrained(model_out_dir)
#                 config.save_pretrained(model_out_dir)
            
#             if budget > 0:
#                 unlabelled_dataset = load_and_cache_examples_by_index(cfg , unlabelled_index_list , features)
#                 unlabelled_sampler = SequentialSampler(unlabelled_dataset)
#                 unlabelled_dataloader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                
#                 for i in range(num_models):
#                     print("--------------- get the entropy of the unlabelled data ---------------")
#                     model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(i)  , cfg.THIEF.MODEL_TYPE)
#                     model = AutoModelForSequenceClassification.from_pretrained(model_out_dir, num_labels=cfg.VICTIM.NUM_LABELS)
#                     tokenizer = AutoTokenizer.from_pretrained(model_out_dir)
#                     config = AutoConfig.from_pretrained(model_out_dir)
#                     model.to(device)
#                     # index_to_entropy[i] = get_entropy(cfg , model , tokenizer , config , n_gpu , device , unlabelled_dataloader)
#                     index_to_label_prob[i] = get_label_probs(cfg , model , tokenizer , config , n_gpu , device , unlabelled_dataloader)
            
#                 unlabelled_index_list = get_index_by_vote(index_to_label_prob)

#             if budget >= 1000:
#                 labelled_index_list.extend(unlabelled_index_list[:1000])
#                 unlabelled_index_list = unlabelled_index_list[1000:]
#                 budget = budget - 1000
#             elif budget == 0:
#                 break
#             elif budget >= len(unlabelled_index_list) & budget < 1000:
#                 budget = len(unlabelled_index_list)
#                 labelled_index_list.extend(unlabelled_index_list[:budget])  
#                 unlabelled_index_list = unlabelled_index_list[budget:]
#                 budget = -1
#             else:
#                 labelled_index_list.extend(unlabelled_index_list[:budget])
#                 unlabelled_index_list = unlabelled_index_list[budget:]
#                 budget = -1
#             print("------------------- budget left -",budget, "-------------------")
#             cycle_num = cycle_num + 1


#         with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
#             for i in range(10):
#                 for key in validation_accuracy_dict_all_models[i]:
#                     f.write("model {} cycle {} validation accuracy: {} ".format(i, key, validation_accuracy_dict_all_models[i][key]))

#     if cfg.THIEF.DO_TESTING == True:
#           # get all folders in model_out_dir
#         for i in range(num_models):
#                 accuracy_list = {}
#                 model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(i) )
#                 # model = AutoModelForSequenceClassification.from_pretrained(model_out_dir, num_labels=cfg.VICTIM.NUM_LABELS)
#                 # tokenizer = AutoTokenizer.from_pretrained(model_out_dir)
#                 # config = AutoConfig.from_pretrained(model_out_dir)
#                 # model.to(device)
#                 model_folders = os.listdir(model_out_dir)
#                 # sort the model folders alphabetically
#                 model_folders.sort()
#                 model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
#                 print(model_folders)
#                 for model_folder in model_folders:
#                     print(model_folder)
#                     model = AutoModelForSequenceClassification.from_pretrained(model_folder)
#                     tokenizer = AutoTokenizer.from_pretrained(model_folder)
#                     config = AutoConfig.from_pretrained(model_folder)
#                     model.to(device)
#                     victim_name = cfg.VICTIM.DATASET
#                     Dataset_list = Dataset.__dict__.keys()
#                     if victim_name not in Dataset_list:
#                         raise ValueError('Dataset not found. Valid arguments = {}'.format(Dataset_list))
#                     Dataset_victim = Dataset.__dict__[victim_name]
#                     Dataset_victim = Dataset_victim(cfg.VICTIM.DATASET , cfg.VICTIM.DATA_DIR, cfg.VICTIM.NUM_LABELS , label_probs = False, Mode = cfg.VICTIM.DATA_MODE , args = cfg)
#                     if cfg.VICTIM.DATA_MODE == "train":
#                         victim_dataset = Dataset_victim.get_train_examples()
#                     if cfg.VICTIM.DATASET == 'sst2':
#                         result = evaluate_thief(cfg, Dataset_victim, model, tokenizer, config, n_gpu , device, split='dev')
#                     else:
#                         result = evaluate_thief(cfg, Dataset_victim, model, tokenizer, config, n_gpu , device, split='test')
#                     accuracy_list[model_folder] = result
#                 print("accuracy list", accuracy_list)
#                 # save the accuracy list to txt file
#                 with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
#                     for key, value in accuracy_list.items():
#                         f.write("%s,%s" % (key, value))
#                         f.write("\n")


# def Unlimited_Budget(cfg : CfgNode , victim_model , victim_tokenizer , victim_config):
    thief_model , thief_tokenizer , thief_config = load_untrained_model(cfg)
    thief_dataset = load_dataset_thief(cfg)