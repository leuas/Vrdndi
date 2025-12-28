"""
ProductiveModel and HybridProductiveModel Architecture


This module defines the core model class, `ProductiveModel` `HybridProductiveModel`, which 
encapsulates the forward architecture. It is designed to process video data by 
concatenating media metadata with pre-computed (offline) encoded tensors.

Key Components:
    1.  **ProductiveModel (Base):** The base architecture designed to process 
        raw media data.
    2.  **HybridProductiveModel (Primary):** A subclass that predicts the productive rate 
        and interest rate for media data with pre-computed encoded tensors
        representing app sequences

Usage:
    For most inference tasks, `HybridProductiveModel` is the primary class 
    as it incorporates the full feature set. Use `ProductiveModel` only if you 
    intend to subclass the architecture.


Classes:
    - ProductiveModel: Basic model
    - HybridProductiveModel: Child class of productive model. Using offline app sequence as feature.


"""
import logging
import torch
import pprint
import pandas as pd

from typing import Optional
from transformers import AutoModel

from torch import nn
from sentence_transformers import SentenceTransformer

from peft import LoraConfig,get_peft_model

from src.utils.ops import pad_aw_sequence
from src.config import DEVICE,ProductiveModelConfig,HybridProductiveModelConfig

from src.models.components import ResBlock,ConditionAwareSequential,Permute,SwiGLU

class ProductiveModel(nn.Module):
    '''
    Base architecture for processing media metadata features.

    It encodes raw media attributes (such as textual titles 
    and numerical duration) and predicts the interest and
    productive rate.

    Args:
        config (ProductiveModelConfig): The configuration of the model

    
    '''

    def __init__(self,config:ProductiveModelConfig ):
        super().__init__()

        self.config=config

        self.bge=AutoModel.from_pretrained(self.config.model_name)

        self.bge_feature_size=self.bge.config.hidden_size #1024
        
        if self.config.use_lora:
            self._enable_lora()

        

        #plus one for duration
        self.productive_layer=nn.Sequential(
            nn.Linear(self.bge_feature_size+1,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,self.config.productive_out_feature)
        ) 

        self.interest_layer=nn.Sequential(
            nn.Linear(self.bge_feature_size+1,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,self.config.interest_out_feature)
        ) 
        


    def _enable_lora(self)->None:
        '''enable lora to train the model'''

        config=LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=0.1,
            bias='none'
        )

        lora_model=get_peft_model(self.bge,config)
        
        lora_model.print_trainable_parameters()

        self.model=lora_model

        logging.info('lora enabled!')
        
    def _mean_pool(self,last_hidden_state:torch.Tensor,attention_mask:torch.Tensor):
        '''apply mean pool on the model output'''


        unsqueezed_attention_mask=attention_mask.unsqueeze(-1).float() #add one more dimension to match the token_size
        
        masked_output=torch.sum(last_hidden_state*unsqueezed_attention_mask,1)

        real_token_num=torch.clamp(torch.sum(attention_mask,1),min=1e-9).unsqueeze(1) #in case there's a batch of empty tokens

    
        mean_pooled=masked_output/real_token_num

        return mean_pooled


    def _add_duration(self,duration:torch.Tensor,model_tensor:torch.Tensor) ->torch.Tensor:
        '''add duration to the model output sequence'''

        reshaped_duration=duration.unsqueeze(1)
        
        combined_feature=torch.cat((model_tensor,reshaped_duration),dim=1)

        return combined_feature
    
    def _output_custom_layer(self,tensor_feature):
        '''pass the tensor to the productive and interest layer'''

        prodc_ouput=self.productive_layer(tensor_feature)
        interest_ouput=self.interest_layer(tensor_feature)

        return {'productive_rate': prodc_ouput ,'interest':interest_ouput}


    def _move_batch_to_device(self,batch):
        '''Helper function: move the element in the batch to device'''


        new_batch={}

        for key,value in batch.items():
            if isinstance(value,torch.Tensor):
                new_batch[key]=value.to(DEVICE)

            else:
                new_batch[key]=value

        
        return new_batch



    def forward(self,input_ids:torch.LongTensor, attention_mask:torch.Tensor,duration:torch.Tensor,**kargs) -> dict:
        '''forward pass of the model
         
        Args:
            inputs_ids: (batch_size, sequence_length)
            attention_mask: (batch_size,sequence_length)
            duration: the duration of the input video in seconds; (batch_size,)
        '''


        outputs= self.bge(input_ids=input_ids,attention_mask=attention_mask)
        

        last_hidden_state=outputs.last_hidden_state

        mean_pooled=self._mean_pool(last_hidden_state,attention_mask)

        
        combined_feature=self._add_duration(duration,mean_pooled)

    
        return  self._output_custom_layer(combined_feature)
    

    def predict_step(self,batch) ->dict:
        '''predict one step using model forward '''


        new_batch=self._move_batch_to_device(batch)

        return self(**new_batch)
    







class HybridProductiveModel(ProductiveModel):
    '''
    Model class that uses offline ActivityWatch (AW) sequences with text to
    predict 'interest' and 'productive' scores.

    This class extends the base architecture by combining offline app sequence tensor
    directly into the transformer's input sequence.

    Architecture / Forward Pass:
        1.  Three residual blocks compress AW sequence into one single token.
        2.  BGE-M3 embedding layer embed the input text (`input_ids`)
        3.  Concatenated the compressed token (from step 1) with the text embeddings (from step 2). 
        4.  The BGE-M3 encoder processes the sequence of tokens
        5.  A mean pooling operation is applied to the encoder's output 
            to derive a global feature vector.
        6.  The meanpooled output is passed through separate heads to predict 
            the 'interest' and 'productive' scores.

    Args:
        config (HybridProductiveModelConfig): Model configuration.
    
    '''

    def __init__(self,config:HybridProductiveModelConfig):

        self.config=config
    
        super().__init__(self.config)


        self.embed_out_feature =384 #The output token size of text_encoder (Sentence Transformer)

        self.num_projector=nn.Linear(self.config.num_in_feature,self.config.num_out_feature)

        self.combined_feature=self.config.num_out_feature+self.embed_out_feature


        self.main_feature_embed_layer=self.bge.embeddings
        

        self.sequence_compressor=ConditionAwareSequential(
            Permute(), # -> (Batch, Token_size, Sequence_size)
            ResBlock(self.combined_feature,894,kernel_size=7,stride=2,cond_dim=self.config.cond_dim),
            ResBlock(894,1024,kernel_size=5,stride=2,cond_dim=self.config.cond_dim),
            ResBlock(1024,1024,kernel_size=3,stride=1,cond_dim=self.config.cond_dim),
            nn.AdaptiveAvgPool1d(1),
            Permute(),
        )

        self.productive_layer=nn.Sequential(
            nn.Linear(self.bge_feature_size,256),
            nn.ReLU(),
            nn.Dropout(self.config.productive_output_layer_dropout),
            nn.Linear(256,self.config.productive_out_feature)
        )

        self.interest_layer=nn.Sequential(
            SwiGLU(self.bge_feature_size,256),
            nn.Dropout(self.config.interest_output_layer_dropout),
            nn.Linear(256,self.config.interest_out_feature)
        )


    def _pad_aw_sequence(self,aw_num_series:pd.Series,aw_text_series:pd.Series,**kwargs) ->tuple[torch.Tensor,torch.Tensor]:
        ''' Project the numerical tensor and concatenate each numerical tensor and text token tensor from aw_num_series and aw_text_series 

            Args:
                aw_text_tensor(pd.Series):
                A pd.Series contain: 
                    either a torch.Tensor contain aw_text_tensor(a sequence of encoded text data of each aw events),
                    or -100, 
                    depending on whether or not current predicting value has aw_data (i.e. predicting productive_rate)
                aw_num_tensor(pd.Series): 
                    A pd.Series contains: 
                        either a torch.Tensor contain aw_num_tensor(a sequence of numerical data of each aw events ),
                        or -100, 
                        depending on whether or not current predicting value has aw_data(i.e. predicting productive_rate)

            Returns:
                tuple[torch.Tensor, torch.Tensor]
                For more detail, you could see the Returns string of pad_aw_sequence
        
        
        '''

        projected_num_series=aw_num_series.apply(lambda onedata: self.num_projector(onedata.to(DEVICE)) if isinstance(onedata,torch.Tensor) else -100)
        
        combined_tensor_list=[]
        for num_tensor,text_tensor in zip(projected_num_series,aw_text_series):

            if isinstance(num_tensor,torch.Tensor) and isinstance(text_tensor,torch.Tensor):
                combined_tensor=torch.cat((text_tensor.to(DEVICE),num_tensor),dim=1) #shape: (seq_size, token_size)
                combined_tensor_list.append(combined_tensor)

            else:
                combined_tensor_list.append(-100)

        combined_tensor_series=pd.Series(combined_tensor_list)

        aw_tensor,aw_attention_mask=pad_aw_sequence(combined_tensor_series,aw_tensor_feature_size=self.combined_feature)

        return aw_tensor,aw_attention_mask




    def _featurize_aw_sequence(self,aw_tensor:torch.Tensor,aw_attention_mask:torch.Tensor,duration:torch.Tensor) ->tuple[torch.Tensor,torch.Tensor]:
        ''' Use convolution to compress the AW sequence to several token'''



        compressed_embed=self.sequence_compressor(aw_tensor,duration)

        #there's only one sequence at the end
        compressed_mask=torch.ones((compressed_embed.shape[0],1),device=DEVICE)



        return compressed_embed,compressed_mask



    def forward(self,input_ids:torch.LongTensor, attention_mask:torch.Tensor,duration:torch.Tensor,aw_tensor:torch.Tensor,aw_attention_mask:torch.Tensor,**kwargs) -> dict:
        '''Forward pass of the model

        Steps:
            1, Use residual block to compress and encode the AW sequence data into one token
            2, Embed the input_ids by using BGE-m3's ( Main model's ) embedding layer
            3, Combine the embeded sequence token from step 2 with the token from step one, do the same thing with attention mask
            4, Extend the attention mask for multi head attention
            5, Use BGE-m3's encoder part to encode the tensor from step 3
            6, Apply mean pool to encdoer's output t
            7, Process the output use interest layer and productive layer 
         
        Args:
            inputs_ids(torch.LongTensor): Shape: (batch_size, sequence_length)
            attention_mask(torch.Tensor): Shape: (batch_size,sequence_length)
            duration(torch.Tensor): The duration of the input video(or other media) in seconds; (batch_size,)
            aw_tensor:{torch.Tensor}: The padded AW data tensor from pad_aw_sequence
            aw_attention_mask:{torch.Tensor}: The padded AW attention mask tensor from pad_aw_sequence
            

        Returns:
            Type: {dict}
            For more details, you could see Returns string of _output_custom_layer() in ProductiveModel class
        '''
        
        compressed_embed,compressed_mask=self._featurize_aw_sequence(aw_tensor,aw_attention_mask,duration)
        
        
        main_feature_embed=self.main_feature_embed_layer(input_ids=input_ids)


        combined_embed=torch.cat([main_feature_embed,compressed_embed],dim=1)
        combined_mask=torch.cat([attention_mask,compressed_mask],dim=1)
        
        
        #Creates an "extended" mask for multi-head attention
        #1, Transform the shape from (Batch, Sequence) to (Batch, NumHead, SequenceLength, Sequence)
        #2, Converts binary flags (1/0) to additive penalties (0/-10000) so padding token could be ignored by attention softmax
        extended_attention_mask=self.bge.get_extended_attention_mask(
            attention_mask=combined_mask,
            input_shape=combined_mask.shape
        )

        encoder_output=self.bge.encoder(
            hidden_states=combined_embed,
            attention_mask=extended_attention_mask,
        )

        #mean_pooled Shape(Batch,Token_size ) 
        mean_pooled=self._mean_pool(encoder_output.last_hidden_state,combined_mask)
        

        return self._output_custom_layer(mean_pooled)


    def predict_step(self,batch) ->dict:
        '''predict one step using model forward '''


        new_batch=self._move_batch_to_device(batch)

        aw_tensor,aw_attention_mask=self._pad_aw_sequence(**new_batch)

        new_batch['aw_tensor']=aw_tensor
        new_batch['aw_attention_mask']=aw_attention_mask

        return self(**new_batch)












        






    




