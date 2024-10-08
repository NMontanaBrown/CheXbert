from typing import Union
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import chexbert.bert_tokenizer as bert_tokenizer
from torch.utils.data import Dataset

class UnlabeledDataset(Dataset):
        """The dataset to contain report impressions without any labels."""
        
        def __init__(self, csv_path:Union[str, pd.DataFrame],
                     column:str="Report Impression"):
                """ Initialize the dataset object
                @param csv_path (string): path to the csv file containing the reports. It
                                          should have a column named "column".
                @param column (string): the name of the column in the csv file that contains
                                            the report impressions.
                """
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                impressions = bert_tokenizer.get_impressions_from_csv(csv_path, column)
                self.encoded_imp = bert_tokenizer.tokenize(impressions, tokenizer)

        def __len__(self):
                """Compute the length of the dataset

                @return (int): size of the dataframe
                """
                return len(self.encoded_imp)

        def __getitem__(self, idx):
                """ Functionality to index into the dataset
                @param idx (int): Integer index into the dataset

                @return (dictionary): Has keys 'imp', 'label' and 'len'. The value of 'imp' is
                                      a LongTensor of an encoded impression. The value of 'label'
                                      is a LongTensor containing the labels and 'the value of
                                      'len' is an integer representing the length of imp's value
                """
                if torch.is_tensor(idx):
                        idx = idx.tolist()
                imp = self.encoded_imp[idx]
                imp = torch.LongTensor(imp)
                return {"imp": imp, "len": imp.shape[0]}
