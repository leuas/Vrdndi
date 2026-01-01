

from typing import Iterator
from torch.utils.data import IterableDataset
from datasets import load_dataset

class RecursiveTrainingData(IterableDataset):
    '''
    The dataset class for recursive BGE-M3 dataset
    
    '''
    def __init__(self,buffer_size:int=10000,seed:int=42) -> None:
        super().__init__()

        self.dataset=load_dataset("c4", "en","fr","zh","jp",split="train", streaming=True)\
        .shuffle(buffer_size=buffer_size,seed=seed)

    

    def __iter__(self) -> Iterator:
        
        for row in self.dataset:
            text = row['text']
            
            if len(text) > 50 and len(text) < 2000:
                yield text




