
from transformers import AutoTokenizer
from typing import Iterator, Literal
from torch.utils.data import IterableDataset
from datasets import load_dataset,interleave_datasets, get_dataset_config_names

from src.config import RecursiveBGEConfig

class RecursiveTrainingData(IterableDataset):
    '''
    The dataset class for recursive BGE-M3 dataset
    
    '''
    def __init__(self,config:RecursiveBGEConfig,dataset_path:str="uonlp/CulturaX",split:Literal['train','validation']='train') -> None:
        super().__init__()

        self.config=config
        self.split=split

        if not self.config.debug_mode:
            self.dataset=self._load_dataset(dataset_path)
        else:
            self.dataset=load_dataset(dataset_path,name='en',split='train',streaming=True)

        self.dataset = self.dataset.shuffle(buffer_size=self.config.buffer_size,seed=self.config.seed)

        if split == "train":
            self.dataset= self.dataset.skip(self.config.eval_set_size)
            
        else:
            self.dataset = self.dataset.take(self.config.eval_set_size)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ori_model_name)
    
    def _load_dataset(self,ori_dataset_path:str):
        '''load different lanaguage from the dataset'''

        all_languages=get_dataset_config_names(ori_dataset_path)

        datasets_list=[]

        for lang in list(all_languages):
            try:
                # Load ONE language at a time
                ds = load_dataset(
                    ori_dataset_path, 
                    name=lang,
                    split=self.split,
                    streaming=True
                )
                datasets_list.append(ds)
            except Exception as e:
                print(f"Warning: Could not load language '{lang}': {e}")

        # 3. Combine them into one single dataset
        if datasets_list:
            whole_dataset = interleave_datasets(datasets_list, seed=self.config.seed)
        else:
            raise ValueError("No languages were loaded successfully.")
        
        return whole_dataset


    def __iter__(self) -> Iterator:
        
        for row in self.dataset:
            text = row['text']

            encodings=self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.config.max_lengh
            )
            
            if 50 <= len(text) <= 2000:
                yield {
                    'input_ids':encodings['input_ids'],
                    'attention_mask':encodings['attention_mask']
                }




