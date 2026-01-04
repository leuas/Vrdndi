

from typing import Iterator, Literal
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from torch.utils.data import IterableDataset
from datasets import load_dataset,interleave_datasets, get_dataset_config_names


from src.config import RecursiveBGEConfig

class RecursiveTrainingData(IterableDataset):
    '''
    The dataset class for recursive BGE-M3 dataset
    
    '''
    def __init__(self,config:RecursiveBGEConfig,dataset_path:str="allenai/c4",split:Literal['train','validation']='train') -> None:
        super().__init__()

        self.config=config
        self.split=split
        self.dataset_path=dataset_path

        if not self.config.debug_mode:
            self.dataset=self._load_datasets()
        else:
            self.dataset=load_dataset(dataset_path,name='en',split='train',streaming=True)

        self.dataset = self.dataset.shuffle(buffer_size=self.config.buffer_size,seed=self.config.seed)

        if split == "train":
            self.dataset= self.dataset.skip(self.config.eval_set_size)
            
        else:
            self.dataset = self.dataset.take(self.config.eval_set_size)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ori_model_name)


    def _load_language_dataset(self,lang:str) ->tuple[IterableDataset,str]:
        '''Helper function: load one lanaguage from a dataset'''
        
        try:

            ds=load_dataset(
                    self.dataset_path, 
                    name=lang,
                    split=self.split,
                    streaming=True,
                )
            
            next(iter(ds))
            return (ds.select_columns(['text']),lang)
            
        except Exception as e:
            print(f'ERROR: FAILED {lang}: {str(e)}')
            return None
    
    def _load_datasets(self):
        '''load different lanaguage from the dataset'''

        all_languages=get_dataset_config_names(self.dataset_path)
        

        with ThreadPoolExecutor(max_workers=self.config.thread_pool_max_worker) as ex:
            output=ex.map(self._load_language_dataset,all_languages)

        
        valid_output=[o for o in output if o is not None]

        datasets_list=[output[0] for output in valid_output]

        loaded_languages= [output[1] for output in valid_output] 

        if not datasets_list:
            raise ValueError("No languages of dataset loaded successfully.")
        
        has_en='en' in loaded_languages

        non_en_prob=0.5/(len(loaded_languages)-1)\
            if has_en and len(loaded_languages) > 1 else 1.0/len(loaded_languages)
        
        probabilities = [0.5 if lang == 'en' else non_en_prob for lang in loaded_languages]
        
        
        # 3. Combine them into one single dataset
        whole_dataset = interleave_datasets(
            datasets_list,
            seed=self.config.seed,
            stopping_strategy="all_exhausted",
            probabilities=probabilities
            )
        
        return whole_dataset


    def __iter__(self) -> Iterator:
        
        for row in self.dataset:

            encodings=self.tokenizer(
            row,
            truncation=True,
            padding=False,
            max_length=self.config.max_lengh
            )
            
            if 50 <= len(row) <= 2000:
                yield {
                    'input_ids':encodings['input_ids'],
                    'attention_mask':encodings['attention_mask']
                }




