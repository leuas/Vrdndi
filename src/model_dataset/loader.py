'''conta dataloader function or other functin for loading data'''

import pandas as pd
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader,WeightedRandomSampler

from src.model_dataset.productive import ProductiveData,HybirdProductiveData

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

from src.utils.ops import split_xy
from src.config import HybirdProductiveModelConfig

from src.path import TRAIN_DATA_PATH,INFERENCE_DATA_PATH


def seed_worker(worker_id):
    '''setup the random seed for worker in dataloader '''
    worker_seed=torch.initial_seed() %2**32
    
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    


class ProductiveLoader:
    '''Dataloader part for Productive Model'''

    def __init__(self,seed:int=42) -> None:
        
        self.g=torch.Generator().manual_seed(seed)

    def dataloader(self,data:pd.DataFrame,batch_size:int,shuffle:bool=True) -> DataLoader:
        '''convert dataset to DataLoader'''

        dataset=ProductiveData(data,max_length=4096)
        tokenizer=dataset.tokenizer
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

        dataloader=DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            num_workers = 0,
            generator=self.g,
            worker_init_fn=seed_worker
        )

        return dataloader
    
    def train_dataloader(self,train_set:pd.DataFrame,batch_size:int,shuffle:bool=True) -> DataLoader:
        '''dataloader for training set'''


        return self.dataloader(train_set,batch_size,shuffle)


class HybirdProductiveLoader:
    '''Data loader part for Hyvird Productive Model'''

    def __init__(self,config:HybirdProductiveModelConfig,seed:int=42) -> None:


        self.g=torch.Generator().manual_seed(seed)

        self.config=config

        self.tokenizer=AutoTokenizer.from_pretrained('BAAI/bge-m3',local_files_only=True)



    def hybird_data_collator(self,dataset:list) ->dict:
        '''combine the aw data collator with other feature's DataCollatorWithPadding
        Args:
            dataset(list): including input ids, attention mask, duration and timestamp and ground truth(if have any)
         
         
        '''

        aw_text_tensor_list=[]
        aw_num_tensor_list=[]
        videoid_list=[]
        main_items_list=[] #including input ids, attention mask, duration and ground truth


        main_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)

        for item in dataset:
            aw_text_tensor_list.append(item.pop('aw_text_tensor'))
            aw_num_tensor_list.append(item.pop('aw_num_tensor'))
            videoid_list.append(item.pop('videoId'))

            main_items_list.append(item)

        final_batch=main_collator(main_items_list)

        #there''s some data which doesn't have aw_data, in that case, both would be -100,
        #so here we still can't stack the tensor together(to have a batch size)
        text_tensor_series=pd.Series(aw_text_tensor_list)
        num_tensor_series=pd.Series(aw_num_tensor_list)
        videoid_series=pd.Series(videoid_list)


        final_batch['aw_text_series']=text_tensor_series
        final_batch['aw_num_series']=num_tensor_series
        final_batch['videoId_series']=videoid_series

        return final_batch
    
    def get_weighted_random_sampler(self,y:pd.DataFrame) ->WeightedRandomSampler:
        '''Calculate the sample weight and get the WeightedTrandomSampler for productive data'''

        productive_class_couts=y['productive_rate'].value_counts()
        

        count_dict=productive_class_couts.to_dict()
        print(count_dict)

        total_count=sum(count_dict.values())

        interest_ratio=self.config.sampler_interest_ratio
        productive_ratio=self.config.sampler_productive_ratio

        #In productive_rate column, if there's no productive rate(-100), it's simply interest
        interest_sample_weight=interest_ratio*total_count/count_dict[-100]
        productive_sample_weight=productive_ratio*total_count/(count_dict[0]+count_dict[1])

        print(f'Productive_sample weight: {productive_sample_weight} ;Interest_sample_weight: {interest_sample_weight}')

        weight_map={
            -100:interest_sample_weight,
            1:productive_sample_weight,
            0:productive_sample_weight
        }

        self.config.wandb_config['productive_class_cout']=count_dict
        self.config.wandb_config['productive_sample_weight']=productive_sample_weight
        self.config.wandb_config['interest_sample_weight']=interest_sample_weight        
        
        sample_weight_series=y['productive_rate'].map(weight_map)

        sample_weight_tensor=torch.DoubleTensor(sample_weight_series.values)

        sampler=WeightedRandomSampler(
            weights=sample_weight_tensor,
            num_samples=len(sample_weight_tensor),
            replacement=True,
        )

        print('WeightRandomSampler created')

        return sampler
    



    def train_dataloader(self,train_set:pd.DataFrame,batch_size:int,shuffle:bool=False) -> DataLoader:
        '''the dataloader function for training set'''



        _,y=split_xy(data=train_set)

        sampler=self.get_weighted_random_sampler(y=y)

        data=HybirdProductiveData(train_set,path=TRAIN_DATA_PATH,max_length=self.config.max_length)
        

        dataloader=DataLoader(
            data,
            batch_size = batch_size,
            shuffle=shuffle,
            collate_fn=self.hybird_data_collator,
            num_workers = self.config.train_num_workers,
            generator=self.g,
            sampler=sampler,
            drop_last=True,
            worker_init_fn=seed_worker
        )

        return dataloader


    def dataloader(self,data:pd.DataFrame,batch_size:int,shuffle:bool=False,path:Path|None=None) -> DataLoader:
        '''convert dataset to DataLoader
            Args:
                data (pd.DataFrame): The source DataFrame contain the Input (and ground Truth optional) of the model
                batch_size (int): The number of samples per batch
                shuffle (bool, optional): whether to shuffle the data when load to dataloader. Defaulting to False
                path (Path, optional): The path to the directory containing pre-computed tensor required by (i.e. AW sequence tensor)
        
        
        '''

        if path is None:
            path=TRAIN_DATA_PATH
            print('Warning: Argument path is not specified. Defaulting to TRAIN_DATA_PATH  ')

        dataset=HybirdProductiveData(data,path=path,max_length=self.config.max_length)


        dataloader=DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            collate_fn=self.hybird_data_collator,
            num_workers = self.config.eval_test_num_workers,
            generator=self.g,
            worker_init_fn=seed_worker
        )


        return dataloader
        
        

    


