# coding=utf-8

"""
Function to embed a dataset of reports
"""

from typing import Union, List
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import chexbert.utils as utils
from chexbert.models.bert_labeler import bert_embedder
from chexbert.label import load_unlabeled_data
from collections import OrderedDict
from tqdm import tqdm

    
def embed(checkpoint_path:str,
          csv_path:Union[str, pd.DataFrame],
          column:str="Report Impression",
          index_state:Union[int, List[int]]=0,
          batch_size:int=16):
    """
    Embeds a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string or pd.DataFrame): location of csv with reports,
                                                or the dataframe itself.
    @param column (string): column name of reports in csv
    @param index_state (int or List[int]): indices of the states to embed.
                                           Default behaviour is to use the CLS token at index 0.
    @param batch_size (int): batch size for the embedding
    @returns y_pred (List[np.array]): Embeddings for each report
    """
    ld = load_unlabeled_data(csv_path, column=column)
    
    model = bert_embedder(index_state=index_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            # don't load linear heads into the model, as these are not
            # created (or needed) for the embedder.
            if "linear_heads" in k:
                continue
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if "linear_heads" in k:
                continue
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    print("\nBegin report impression embedding. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % batch_size)
    y_pred = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(ld)):
            batch = data['imp'] # (batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            attn_mask = utils.generate_attention_masks(batch, src_len, device)
            out = model(batch, attn_mask) # (batch_size, len(index_state), hidden_size)
            y_pred.append(out.detach().cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred # (len(ld), len(index_state), hidden_size)
