

from typing import Iterator, Literal
from torch.utils.data import IterableDataset
from datasets import load_dataset,interleave_datasets, get_dataset_config_names

class RecursiveTrainingData(IterableDataset):
    '''
    The dataset class for recursive BGE-M3 dataset
    
    '''
    def __init__(self,buffer_size:int=10000,seed:int=42,split:Literal['train','validation']='train') -> None:
        super().__init__()

        all_languages=get_dataset_config_names('mc4')

        languages_to_load=[lang for lang in all_languages]
    

        self.dataset=load_dataset("mc4",name=languages_to_load,split=split, streaming=True)

        # Only shuffle training data.
        if split == "train":
            self.hf_dataset = self.hf_dataset.shuffle(buffer_size=buffer_size,seed=seed)
    

    def __iter__(self) -> Iterator:
        
        for row in self.dataset:
            text = row['text']
            
            if len(text) > 50 and len(text) < 2000:
                yield text




