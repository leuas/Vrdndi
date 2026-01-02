'''the recursive BGE-M3 model'''

import logging
import torch

import torch.nn as nn
from src.models.components import RecursiveACTLayer
from transformers import AutoModel, AutoConfig

from src.config import DEVICE

class DistillRecursiveModel(nn.Module):
    '''recursive small BGE-M3
    Args:
        model_name(str): The original model's name that you used for distill.
            Default to "BAAI/bge-m3"

        max_steps (int): THe maximal number of recursive layer that the model can use

        init_layer_index(int): THe index of pretrained layer you want to use in original model
    
    
    '''

    def __init__(self,model_name:str="BAAI/bge-m3",max_steps:int=36,init_layer_index:int=0) -> None:
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        hidden_dim = self.config.hidden_size

        original_model = AutoModel.from_pretrained(model_name).cpu()
        pretrained_block = original_model.encoder.layer[init_layer_index]

        self.embeddings = original_model.embeddings

        self.recursive_block = RecursiveACTLayer(hidden_dim, max_steps=max_steps, pretrained_layer=pretrained_block)

        del original_model


    def forward(self, input_ids:torch.LongTensor,attention_mask:torch.LongTensor,tau:float=0.01,**kargs) ->tuple[torch.Tensor,torch.Tensor]:
        '''
        forward process for distill model
        
        '''

        embedded_tensor=self.embeddings(input_ids)

        encoded_tensor,step_cost=self.recursive_block(embedded_tensor,attention_mask,tau)

        sentence_embedding=torch.nn.functional.normalize(encoded_tensor,dim=2) #normalize the hidden_dim


        return sentence_embedding,step_cost
    
    def _move_batch_to_device(self,batch):
        '''Helper function: move the element in the batch to device'''


        new_batch={}

        for key,value in batch.items():
            if isinstance(value,torch.Tensor):
                new_batch[key]=value.to(DEVICE)

            else:
                new_batch[key]=value

        
        return new_batch

    def predict_step(self,batch) ->dict:
        '''predict one step using model forward '''


        new_batch=self._move_batch_to_device(batch)

        return self(**new_batch)


        





        









