import torch
import torch.nn.functional as F



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
