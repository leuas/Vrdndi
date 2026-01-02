

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

        datasets_list=[]

        for lang in list(all_languages):
            try:
                # Load ONE language at a time
                ds = load_dataset(
                    dataset, 
                    name=lang,
                    split=split,
                    streaming=True
                )
                datasets_list.append(ds)
            except Exception as e:
                print(f"Warning: Could not load language '{lang}': {e}")

        # 3. Combine them into one single dataset
        if datasets_list:
            self.dataset = interleave_datasets(datasets_list, seed=self.config.seed)
        else:
            raise ValueError("No languages were loaded successfully.")
    
        # Only shuffle training data.
        if split == "train":
            self.hf_dataset = self.hf_dataset.shuffle(buffer_size=self.config.buffer_size,seed=self.config.seed)
    

    def __iter__(self) -> Iterator:
        
        for row in self.dataset:
            text = row['text']
            
            if 50 <= len(text) <= 2000:
                yield text




