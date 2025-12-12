
import json
import pandas as pd

import torch

from torch.utils.data import Dataset

from transformers import AutoTokenizer
from pathlib import Path

from src.utils.ops import split_xy


class ProductiveData(Dataset):
    '''Dataset for multi task(productive and interest) model
    Args:
            data{pd.DataFrame}: The data consist model input x and ground truth y
            max_length{int}: It could be up to 8192,in default it's 120 for better performance
    
    '''

    def __init__(self,data:pd.DataFrame, max_length:int=120):
        x,y=split_xy(data)

        self.x=x

        self.y=y
    
        self.tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.max_length=max_length
        

    def __len__(self) ->int:

        return len(self.x)
    

    def __getitem__(self, index:int) ->dict:

        x=self.x.iloc[index]

        title=x['title']
        description=x['description']
        youtuber=x['youtuber']
        duration=x['duration']
        videoid=x['videoId']


        #tokenize the data

        encodings=self.tokenizer(
            title,
            youtuber,
            description,
            truncation=True,
            padding=False,
            max_length=self.max_length
        )


        item={
            'input_ids':torch.tensor(encodings['input_ids'],dtype=torch.long),
            'attention_mask':torch.tensor(encodings['attention_mask'],dtype=torch.long),
            'duration':torch.tensor(duration,dtype=torch.float),
            'videoId':videoid
        }

        if self.y is not None:
            y=self.y.iloc[index]

            item['productive_rate']=torch.tensor(y['productive_rate'],dtype=torch.long)
            item['interest']=torch.tensor(y['interest'],dtype=torch.long)

            
        return item
    

class HybirdProductiveData(ProductiveData):
    '''Dataset for  HybirdProductive model'''

    def __init__(self,data:pd.DataFrame,path:Path, max_length:int=8192):
        
        super().__init__(data,max_length=max_length)

        with open(path/'manifest.json','r',encoding='utf-8') as j:
            self.manifest=json.load(j)

        self.path=path
        


    def __getitem__(self, index:int) ->dict:

        x=self.x.iloc[index]

        title=x['title']
        description=x['description']
        youtuber=x['youtuber']
        duration=x['duration']
        timestamp=x['timestamp']
        videoid=x['videoId']


        if isinstance(timestamp,str) and timestamp!='-100' :
            try:

                file_name=self.manifest[timestamp]
                tensor_dict=torch.load(self.path/file_name)
                
                #aw_text_tensor contain a sequence of category and title meaning vector of each events
                text_tensor=tensor_dict['aw_text_tensor']
                #aw_num_tensor contain a seuqnce of numerical values of each events
                num_tensor=tensor_dict['aw_num_tensor']

            except (KeyError, TypeError) as e :
                text_tensor=-100
                num_tensor=-100#in case it can't find the key in the manifest

                if isinstance(e, KeyError):
                    print(f"WARNING: Expected {timestamp} has a relate file, but its key doesn't exist in manifest. Set tensor to -100(ignore) as fallback")

                else:
                    print(f"WARNING: Expected torch.Tensor in aw tensor file of {timestamp}, get {type(tensor_dict)} instead.\
                           Set tensor to -100 (ignore) as fallback. Consider delete this record in database to improve model performance  ") 

            
        else:
            text_tensor=-100
            num_tensor=-100


        #tokenize the data

        encodings=self.tokenizer(
            title,
            youtuber,
            description,
            truncation=True,
            padding=False,
            max_length=self.max_length
        )


        item={
            'input_ids':torch.tensor(encodings['input_ids'],dtype=torch.long),
            'attention_mask':torch.tensor(encodings['attention_mask'],dtype=torch.long),
            'duration':torch.tensor(duration,dtype=torch.float),
            'aw_text_tensor':text_tensor,
            'aw_num_tensor':num_tensor,
            'videoId':videoid
        }

        if self.y is not None:
            y=self.y.iloc[index]

            item['productive_rate']=torch.tensor(y['productive_rate'],dtype=torch.long)
            item['interest']=torch.tensor(y['interest'],dtype=torch.long)

            
        return item