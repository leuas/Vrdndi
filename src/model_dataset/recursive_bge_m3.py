

from typing import Iterator, Literal
from torch.utils.data import IterableDataset
from datasets import load_dataset,interleave_datasets, get_dataset_config_names

from src.config import RecursiveBGEConfig

class RecursiveTrainingData(IterableDataset):
    '''
    The dataset class for recursive BGE-M3 dataset
    
    '''
    def __init__(self,config:RecursiveBGEConfig,dataset:str="uonlp/CulturaX",split:Literal['train','validation']='train') -> None:
        super().__init__()

        self.config=config

        all_languages=get_dataset_config_names(dataset)

        languages_to_load=[lang for lang in all_languages]
    

        self.dataset=load_dataset(dataset,name=languages_to_load,split=split, streaming=True)

        # Only shuffle training data.
        if split == "train":
            self.hf_dataset = self.hf_dataset.shuffle(buffer_size=self.config.buffer_size,seed=self.config.seed)
    

    def __iter__(self) -> Iterator:
        
        for row in self.dataset:
            text = row['text']
            
            if 50 <= len(text) <= 2000:
                yield text




