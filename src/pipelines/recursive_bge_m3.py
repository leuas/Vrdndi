'''Training part of the Recursive model'''

import torch
import torch.nn as nn
import transformers

from typing import Iterator

from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModel

from torch.optim import AdamW
from torch.utils.data import DataLoader


from src.config import RecursiveBGEConfig,DEVICE
from src.models.recursive_bge_m3 import DistillRecursiveModel
from src.model_dataset.loader import RecursiveDataLoader

class RecursiveBGETraining:
    '''training part of recursive model'''

    def __init__(self,config:RecursiveBGEConfig) -> None:
        

        self.config=config

        self.ori_model=AutoModel.from_pretrained(self.config.ori_model_name).to(DEVICE)

        self.distill_model=DistillRecursiveModel(
            model_name=self.config.ori_model_name,
            max_steps=self.config.max_steps,
            init_layer_index=self.config.init_layer_index
            ).to(DEVICE)
        
        self.optimizer = AdamW(self.distill_model.parameters(), lr=self.config.lr)
        self.loss_fn = nn.CosineEmbeddingLoss()
        
        self.data=RecursiveDataLoader(self.config)


    def _sort_train_data_to_batch(self,data_stream:Iterator[str]):
        '''process the data and sort them to batch '''

        batch_texts = []
        for _ in range(self.config.batch_size):
            try:
                # Grab next sentence from internet/dataset
                sentence = next(data_stream) 
                batch_texts.append(sentence)
            except StopIteration:
                break
        
        return batch_texts

    def _tokenize_texts(self,batch_texts:list) ->transformers.tokenization_utils_base.BatchEncoding:
        '''tokenize the text input '''
        output = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=8192, 
                    return_tensors="pt"
                ).to(DEVICE)
        
        return output


    
    def _compute_loss(self,ori_output:torch.Tensor,distill_output:torch.Tensor,step_cost:torch.Tensor) ->torch.Tensor:
        '''compute the Embeddings loss between ori model and distill model'''

        target_ones = ori_output.new_ones(ori_output.size(0))
        similarity_loss = self.loss_fn(distill_output, ori_output, target_ones)

        loss = similarity_loss + step_cost

        return loss


    def train(self,save_name:str="bge_recursive.pth"):
        '''start training '''

        print("Starting Training Loop...")
        
        data_stream=self.data.dataloader()

        total_loss = 0
    
        batch_texts=self._sort_train_data_to_batch(data_stream)

        self.ori_model.eval()
        self.distill_model.train()

        for batch in batch_texts:

            with torch.no_grad(): 
                ori_model_output = self.ori_model(batch)
                ori_model_state = ori_model_output.last_hidden_state
                # Normalize Teacher (Crucial!)
                ori_model_state_normed = torch.nn.functional.normalize(ori_model_state, dim=2)
            
            
            distill_output, step_cost = self.distill_model.predict_step(batch)

            loss=self._compute_loss(ori_model_state_normed,distill_output,step_cost)


            # Backpropagation
            self.optimizer.zero_grad() 
            loss.backward()     
            self.optimizer.step()     

            # Logging
            total_loss += loss.item()


    def eval(self):
        '''
        Evaluate the model with some text in dataset
        '''


        




