import torch
import torch.nn.functional as F
from .model import Transformer
import os


def pad_to_max_with_mask(data):
    """
    data: list of tuples (text_src_tensor, text_trg_tensor)
    
    return (b_text_src_tensor, b_text_trg_tensor, b_mask_src_tensor, b_mask_trg_tensor)
    """
    max_len_src = max(text_tuple[0].shape[-1] for text_tuple in data)
    max_len_trg = max(text_tuple[1].shape[-1] for text_tuple in data)

    #  create the mask
    b_text_src_tensor = torch.zeros((len(data), max_len_src), dtype = torch.long)
    b_text_trg_tensor = torch.zeros((len(data), max_len_trg), dtype = torch.long)
    b_mask_src_tensor = torch.ones_like(b_text_src_tensor, dtype = torch.bool)
    b_mask_trg_tensor = torch.ones_like(b_text_trg_tensor, dtype = torch.bool)

    for i in range(len(data)):
        b_text_src_tensor[i] = F.pad(data[i][0], (0, max_len_src - data[i][0].shape[-1]), "constant", 0)
        b_text_trg_tensor[i] = F.pad(data[i][1], (0, max_len_trg - data[i][1].shape[-1]), "constant", 0)
        b_mask_src_tensor[i, :data[i][0].shape[-1]] = 0
        b_mask_trg_tensor[i, :data[i][1].shape[-1]] = 0
    return b_text_src_tensor, b_text_trg_tensor, b_mask_src_tensor, b_mask_trg_tensor




def label_smoothing(tensor, num_classes, num_special_tokens, smoothing_value = 0.1):
    """
    """

    smooth_label = F.one_hot(tensor, num_classes=num_classes).to(torch.float)

    smooth_label[smooth_label == 0] = smoothing_value / (num_classes - num_special_tokens - 1 )
    smooth_label[smooth_label == 1] = 1 - smoothing_value


    return smooth_label




def greedy_decoding(model, text_src, max_output_len = 100, BOS_id = 3, EOS_id = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    text_trg = torch.tensor([[BOS_id]]).to(device)

    while text_trg.shape[-1] < max_output_len:
        #  text_trg = F.pad(text_trg, (0, 1), "constant", 0)
        mask_src = torch.zeros_like(text_src, dtype = torch.bool).to(device)
        mask_trg = torch.zeros_like(text_trg, dtype = torch.bool).to(device)

        predicted_log_distributions = model(text_src, text_trg, mask_src, mask_trg)
        next_log_prediction = predicted_log_distributions[:,-1:,:]
        next_prediction = torch.argmax(next_log_prediction, dim=2)

        print(next_prediction)

        #  text_trg[0][-1] = next_prediction
        text_trg = F.pad(text_trg, (0, 1), "constant", next_prediction)

        if next_prediction == EOS_id:
            break

    return text_trg
