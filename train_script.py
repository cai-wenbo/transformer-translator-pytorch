from numpy import shape
import torch
from torch.cuda import utilization
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.dataset import parallelCorpus
from src.utils import pad_to_max_with_mask, label_smoothing
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.model import Transformer
import json
import os
import argparse




def load_model(model_path_src, training_config):
    '''
    load model
    '''
    model = Transformer(
            model_dimension= training_config["model_dimension"],
            src_vocab_size= training_config["src_vocab_size"],
            trg_vocab_size= training_config["trg_vocab_size"],
            number_of_heads= training_config["number_of_heads"],
            number_of_layers= training_config["number_of_layers"],
            dropout_probability = training_config["dropout_probability"],
            )


    if os.path.exists(model_path_src):
        model_dict = torch.load(training_config['model_path_src'])
        model.load_state_dict(model_dict)

    return model



def load_trails(training_config):
    step_losses = list()
    if os.path.exists(training_config['step_losses_pth']):
        with open(training_config['step_losses_pth'], 'r') as file:
            step_losses = json.load(file)
            file.close()

    train_losses = list()
    if os.path.exists(training_config['train_losses_pth']):
        with open(training_config['train_losses_pth'], 'r') as file:
            train_losses = json.load(file)
            file.close()
    
    eval_losses = list()
    if os.path.exists(training_config['eval_losses_pth']):
        with open(training_config['eval_losses_pth'], 'r') as file:
            eval_losses = json.load(file)
            file.close()

    train_accuracy = list()
    if os.path.exists(training_config['train_accuracy_pth']):
        with open(training_config['train_accuracy_pth'], 'r') as file:
            train_accuracy = json.load(file)
            file.close()
    
    eval_accuracy = list()
    if os.path.exists(training_config['eval_accuracy_pth']):
        with open(training_config['eval_accuracy_pth'], 'r') as file:
            eval_accuracy = json.load(file)
            file.close()


    return step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy


def train_eval_loop(training_config, model, dataloader_train, dataloader_eval, optimizer, creterian, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy, device):
    for epoch in range(training_config['num_of_epochs']):
        loss_sum_train = 0
        err            = 0
        num_tokens     = 0

        model.train()
        #  train loop
        for i, batch in enumerate(dataloader_train):
            batch = tuple(t.to(device) for t in batch)
            b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

            optimizer.zero_grad()
            b_predicted_log_distributions = model(b_text_src, b_text_trg, b_mask_src, b_mask_trg)
            b_smooth_label = label_smoothing(b_text_trg, training_config["trg_vocab_size"], training_config["num_special_tokens_trg"])


            loss = creterian(b_predicted_log_distributions, b_smooth_label)

            loss.backward()
            optimizer.step()
            loss_scalar = loss.item()
            loss_sum_train += loss_scalar
            step_losses.append(loss_scalar)

            b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)
            err += (b_predictions != b_text_trg).sum().item()
            num_tokens += torch.sum(~b_mask_trg).item()



            #  b_predicts = torch.argmax(b_outputs, dim=-1)
            #  correct += (b_predicts == b_labels).sum().item()

        train_loss = loss_sum_train / len(dataloader_train)
        train_losses.append(train_loss)
        train_acc = 1 - err / num_tokens
        train_accuracy.append(train_acc)



        loss_sum_eval = 0
        correct = 0


        model.eval() 
        #  eval_loop
        for i, batch in enumerate(dataloader_eval):
            batch = tuple(t.to(device) for t in batch)
            b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

            with torch.no_grad():
                b_predicted_log_distributions = model(b_text_src, b_text_trg, b_mask_src, b_mask_trg)
                b_smooth_label = label_smoothing(b_text_trg, training_config["trg_vocab_size"], training_config["num_special_tokens_trg"])

                loss = creterian(b_predicted_log_distributions, b_smooth_label)
                loss_scalar = loss.item()
                loss_sum_eval += loss_scalar

                b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)
                err += (b_predictions != b_text_trg).sum().item()
                num_tokens += torch.sum(~b_mask_trg).item()


        eval_loss = loss_sum_eval / len(dataloader_eval)
        eval_losses.append(eval_loss)
        eval_acc = 1 - err / num_tokens
        eval_accuracy.append(eval_acc)



        print(f'Epoch: {epoch+1} \n Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} \ntrain Eval Loss: {eval_loss:.6f}, Eval Acc: {eval_acc:.6f}')



def save_trails(training_config, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy):
    with open(training_config['step_losses_pth'], 'w') as file:
        json.dump(step_losses, file)
        file.close()

    with open(training_config['train_losses_pth'], 'w') as file:
        json.dump(train_losses, file)
        file.close()
    
    with open(training_config['eval_losses_pth'], 'w') as file:
        json.dump(eval_losses, file)
        file.close()

    with open(training_config['train_accuracy_pth'], 'w') as file:
        json.dump(train_accuracy, file)
        file.close()
    
    with open(training_config['eval_accuracy_pth'], 'w') as file:
        json.dump(eval_accuracy, file)
        file.close()



def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    model
    load model and the history
    '''
    model = load_model(training_config['model_path_src'], training_config).to(device)


    #  load the losses history
    step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy = load_trails(training_config)



    '''
    dataloader
    '''
    train_dataset = parallelCorpus(corpus_path_src=training_config["data_path_eval_src"], corpus_path_trg=training_config["data_path_eval_trg"]  , tokenizer_path_src=training_config["tokenizer_path_src"] , tokenizer_path_trg=training_config["tokenizer_path_trg"])
    eval_dataset = parallelCorpus(corpus_path_src=training_config["data_path_eval_src"], corpus_path_trg=training_config["data_path_eval_trg"]  , tokenizer_path_src=training_config["tokenizer_path_src"] , tokenizer_path_trg=training_config["tokenizer_path_trg"])


    dataloader_train = DataLoader(train_dataset, collate_fn=pad_to_max_with_mask, batch_size=5, shuffle=True)
    dataloader_eval = DataLoader(eval_dataset, collate_fn=pad_to_max_with_mask, batch_size=5, shuffle=False)


    '''
    optimizer
    '''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},

        # Filter for parameters which *do* include those.
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # Note - `optimizer_grouped_parameters` only includes the parameter values, not
    # the names.

    optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr = training_config['learning_rate'],
            eps = 1e-8
            )

    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0, # Default value in run_glue.py
            num_training_steps = len(dataloader_train) * training_config['num_of_epochs']
            )




    '''
    creterian
    '''
    creterian = nn.KLDivLoss(reduction='batchmean')


    '''
    train  and validate loops
    '''
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    train_eval_loop(training_config, model, dataloader_train, dataloader_eval, optimizer, creterian, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy, device)


        
    '''    
    save model and data
    '''

    model = model.to('cpu').module
    torch.save(model.state_dict(), training_config['model_path_dst'])

    #  save the loss of the steps
    save_trails(training_config, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs"          , type=int   , help="number of epochs"                                    , default=20)
    parser.add_argument("--batch_size"             , type=int   , help="batch size"                                          , default=512)
    parser.add_argument("--learning_rate"          , type=float , help="learning rate"                                       , default=1e-4)
    parser.add_argument("--weight_decay"           , type=float , help="weight_decay"                                        , default=1e-4)
    parser.add_argument("--dropout_probability"    , type=float , help="dropout prob"                                        , default=1e-1)
    parser.add_argument("--src_vocab_size"         , type=int   , help="src vocab size"                                      , default=5000)
    parser.add_argument("--trg_vocab_size"         , type=int   , help="trg vocab size"                                      , default=5000)
    parser.add_argument("--model_dimension"        , type=int   , help="model dimmention"                                    , default=512)
    parser.add_argument("--number_of_heads"        , type=int   , help="number of heads"                                     , default=8)
    parser.add_argument("--number_of_layers"       , type=int   , help="number of layers"                                    , default=6)
    parser.add_argument("--num_special_tokens_trg" , type=int   , help="number of special tokens of target"                  , default=2)
    parser.add_argument("--num_labels"             , type=int   , help="types of labels"                                     , default=6)
    parser.add_argument("--num_neg"                , type=int   , help="num of neg"                                          , default=2)
    parser.add_argument("--sequence_length"        , type=int   , help="sequence_length"                                     , default=128)
    parser.add_argument("--tokenizer_path_src"     , type=str   , help="the saved tokenizer of src language"                 , default='./data/tokenizer_zh')
    parser.add_argument("--tokenizer_path_trg"     , type=str   , help="the saved tokenizer of trg language"                 , default='./data/tokenizer_en')
    parser.add_argument("--data_path_train_src"    , type=str   , help="the training dataset of src language"                , default='./data/train.zh')
    parser.add_argument("--data_path_train_trg"    , type=str   , help="the training dataset of trg language"                , default='./data/train.en')
    parser.add_argument("--data_path_eval_src"     , type=str   , help="the eval dataset of src language"                    , default='./data/val.zh')
    parser.add_argument("--data_path_eval_trg"     , type=str   , help="the eval dataset of trg language"                    , default='./data/val.en')
    parser.add_argument("--model_path_dst"         , type=str   , help="the directory to save model"                         , default='./saved_models/saved_dict.pth')
    parser.add_argument("--model_path_src"         , type=str   , help="the directory to load model"                         , default='./saved_models/saved_dict.pth')
    parser.add_argument("--step_losses_pth"        , type=str   , help="the path of the json file that saves step losses"    , default='./trails/step_losses.json')
    parser.add_argument("--train_losses_pth"       , type=str   , help="the path of the json file that saves train losses"   , default='./trails/train_losses.json')
    parser.add_argument("--eval_losses_pth"        , type=str   , help="the path of the json file that saves eval losses"    , default='./trails/eval_losses.json')
    parser.add_argument("--train_accuracy_pth"     , type=str   , help="the path of the json file that saves train accuracy" , default='./trails/train_accuracy.json')
    parser.add_argument("--eval_accuracy_pth"      , type=str   , help="the path of the json file that saves eval accuracy"  , default='./trails/eval_accuracy.json')

    
    args = parser.parse_args()

    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    train(training_config)
