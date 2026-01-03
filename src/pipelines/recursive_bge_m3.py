'''Training part of the Recursive model'''

import torch
import torch.nn as nn
import transformers
import wandb

from dataclasses import asdict
from tqdm import tqdm
from itertools import islice
from typing import Iterator

from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModel

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast


from src.config import RecursiveBGEConfig,DEVICE
from src.models.recursive_bge_m3 import DistillRecursiveModel
from src.model_dataset.loader import RecursiveDataLoader

from src.utils.ops import move_batch_to_device,set_random_seed

class RecursiveBGETraining:
    '''training part of recursive model'''

    def __init__(self,config:RecursiveBGEConfig) -> None:
        

        self.config=config

        if self.config.wandb_config.get('TrainingConfig') is None:
            self.config.wandb_config['TrainingConfig']=asdict(self.config)

        self.ori_model=AutoModel.from_pretrained(self.config.ori_model_name).to(DEVICE)
        self.ori_model=self.ori_model.to(torch.bfloat16)

        self.distill_model=DistillRecursiveModel(
            model_name=self.config.ori_model_name,
            max_steps=self.config.max_steps,
            init_layer_index=self.config.init_layer_index
            ).to(DEVICE)
        
        self.optimizer = AdamW(self.distill_model.parameters(), lr=self.config.lr)
        self.loss_fn = nn.CosineEmbeddingLoss()
        
        self.data=RecursiveDataLoader(self.config)
        self.scaler=GradScaler()

    
    def _compute_loss(self,ori_output:torch.Tensor,distill_output:torch.Tensor,step_cost:torch.Tensor) ->torch.Tensor:
        '''compute the Embeddings loss between ori model and distill model'''
        #Pick [batch_size,seq_len] so it could compare the token_size
        target_ones = ori_output.new_ones([ori_output.size(0),ori_output.size(1)])

        distill_output_flat=distill_output.view(-1,distill_output.size(-1))
        ori_output_flat=ori_output.view(-1,ori_output.size(-1))

        target_flat=target_ones.view(-1)
        similarity_loss = self.loss_fn(distill_output_flat, ori_output_flat, target_flat)

        loss = similarity_loss + step_cost

        return loss
    
    def ori_model_predict(self,batch) ->torch.Tensor:
        '''
        use original model to predict current batch
        '''
        new_batch=move_batch_to_device(batch)

        with torch.no_grad(): 
            ori_model_output = self.ori_model(**new_batch)
            ori_model_state = ori_model_output.last_hidden_state
            # Normalize data
            ori_model_state_normed = torch.nn.functional.normalize(ori_model_state, dim=2)

        return ori_model_state_normed



    def train(self,save_name:str="bge_recursive.pth") ->None:
        '''start training '''

        print("Starting Training Loop...")
        
        train_data=self.data.train_dataloader()
        finite_train_data=islice(train_data,self.config.train_batch_num)
        tqdm_data=tqdm(finite_train_data,desc="Runing Training Process")

        total_loss = 0

        self.ori_model.eval()
        self.distill_model.train()

        for i,batch in enumerate(tqdm_data):
            with autocast(device_type=DEVICE.type):
            
                ori_model_output=self.ori_model_predict(batch)
                
                distill_output, step_cost = self.distill_model.predict_step(batch)

                loss=self._compute_loss(ori_model_output,distill_output,step_cost)

                loss=loss/self.config.accumulation_steps

            self.scaler.scale(total_loss).backward()


            if (i+1)%self.config.accumulation_steps==0:

                self.scaler.unscale_(self.optimizer)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Logging
            total_loss += loss.item()

            wandb.log({"Training Loss":loss, "batch_num":i})


    def eval(self) -> None:
        '''
        Evaluate the model with some text in dataset
        '''

        eval_data=self.data.eval_dataloader()

        finite_eval_data=islice(eval_data,self.config.eval_batch_num)
        tqdm_data=tqdm(finite_eval_data,desc="Evaluating Model Performance")
        
        self.ori_model.eval()
        self.distill_model.eval()

        total_loss=0

        for batch in tqdm_data:

            ori_model_output=self.ori_model_predict(batch)
            
            distill_output, step_cost = self.distill_model.predict_step(batch)

            loss=self._compute_loss(ori_model_output,distill_output,step_cost)

            total_loss+=loss.item()


        avg_loss=total_loss/self.config.eval_batch_num

        similarity_score = 1 - avg_loss
    
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Similarity: {similarity_score * 100:.2f}%")

        wandb.log({"Validation Loss":avg_loss, "Similarity":similarity_score*100})



    def epoch_training_loop(self):
        '''train in a epoch loop'''

        for _ in range(self.config.total_epoch):

            self.train()

            self.eval()

    def _set_seed(self,seed:int = 42) -> None:
        '''set the random seed for random,numpy,torch mps,os,sklearn and torch generator'''

        set_random_seed(seed)

        self.config.seed=seed


    def start_train(self,run_name:str):
        '''start the training process'''
        wandb.init(
            project='personal_feed',
            name=run_name,
            config=self.config.wandb_config
            )

        wandb.watch(self.distill_model,log='all',log_freq=10)

        self._set_seed(self.config.seed)

        self.epoch_training_loop()

        wandb.finish()

        

        

        




        

        





        




