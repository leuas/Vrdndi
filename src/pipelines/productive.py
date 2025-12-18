'''the training process of productive model'''
import ast
import argparse
import copy
import logging
import os
import pprint
import wandb
import torch

import pandas as pd
import numpy as np

from typing import TypeVar,Generic
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding

from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss,Module
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from torchmetrics import F1Score,Recall,Precision,MetricCollection

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import KFold

from src.utils.ops import print_parameter_state,data_split,set_random_seed,if_load_model,FocalLoss
from src.utils.data_etl import iso_duration_transform
from src.models.productive import ProductiveModel,HybridProductiveModel

from src.model_dataset.productive import ProductiveData,HybridProductiveData
from src.model_dataset.loader import ProductiveLoader,HybridProductiveLoader

from src.db.database import VrdndiDatabase
from src.config import DEVICE,ProductiveModelConfig,HybridProductiveModelConfig

from src.path import ARTIFACTS_PATH


ConfigType=TypeVar('ConfigType',bound='ProductiveModelConfig')



class ProductiveModelTraining(Generic[ConfigType]):
    '''the class for training and evaluating process'''
    def __init__(self,model:torch.nn.Module|None = None, config:ConfigType|None=None):
        
        self.config=self._setup_config(config)

        self.model=self._setup_model(model)

        if self.config.compile_model:
            self.model=self._initial_model()


        self.interest_loss_fn = None
        self.productive_loss_fn=None


        self.loader= ProductiveLoader(self.config.seed)
        self.optimizer = AdamW(self.model.parameters(), lr = self.config.lr)

        self.db=VrdndiDatabase()

        self.interest_train_metrics=self._get_metrics_bundle()
        self.productive_train_metrics=self._get_metrics_bundle()
        self.interest_val_metrics=self._get_metrics_bundle()
        self.productive_val_metrics=self._get_metrics_bundle()
        

    def _get_metrics_bundle(self) ->MetricCollection:
        '''get a metrics collection from torchmetrics'''

        return MetricCollection({
            'precision':Precision(task='binary'),
            'recall':Recall(task='binary'),
            'f1':F1Score(task='binary'),
        }).to(DEVICE)


    def _setup_model(self,model):
        '''Initial training model and training config'''
        
        if model is None:

            return ProductiveModel(self.config).to(device=DEVICE)
        

        return model
        
    def _initial_model(self):
        '''Compile the model '''

        return torch.compile(self.model,mode='reduce-overhead')


    def _setup_config(self,config:ConfigType|None) ->ProductiveModelConfig:
        '''Initial training config'''

        if config is None:
            logging.info('ProductiveModel is using default config')
            return ProductiveModelConfig()
            

        return config
    


    def _calc_weight(self,input:pd.Series) ->np.ndarray:
        '''calculate weight'''

        mask = input!=self.config.ignore_index
        maksed_y=input[mask]
        
        classes=np.array([0,1])

        weight=compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=maksed_y
        )
        return weight



    def _calculate_interest_weight(self,interest:pd.Series) ->torch.Tensor:
        '''calculate the interest weight '''

        
        weight=self._calc_weight(interest)
        self.config.wandb_config['interest_posweight']=weight

        logging.info("interest weight: ")
        logging.info(weight)

        return torch.tensor(weight,dtype=torch.float32,device=DEVICE)
    
    def _calculate_productive_weight(self,productive:pd.Series) ->torch.Tensor:
        '''calculate the productive weight '''

        weight=self._calc_weight(productive)

        self.config.wandb_config['productive_posweight']=weight
        logging.info("productive weight: ")
        logging.info(weight)

        return torch.tensor(weight,dtype=torch.float32,device=DEVICE)
    
    def _calc_head_loss(self,data:pd.DataFrame) ->tuple[torch.Tensor,torch.Tensor]:
        '''Helper function: calculate the loss for both head (interest and productive_rate)'''

        interest=data['interest']
        productive_rate=data['productive_rate']

        interest_weight=self._calculate_interest_weight(interest)
        productive_weight=self._calculate_productive_weight(productive_rate)

        return interest_weight,productive_weight
    
    def _define_loss_function(self,data:pd.DataFrame) ->None:
        '''define the two head's loss function'''

        interest_weight,productive_weight=self._calc_head_loss(data)
        
        self.interest_loss_fn=CrossEntropyLoss(weight=interest_weight,ignore_index=self.config.ignore_index)
        self.productive_loss_fn=CrossEntropyLoss(weight=productive_weight,ignore_index=self.config.ignore_index)



    def _set_loss_fn(self,data: pd.DataFrame) -> None:
        '''set the loss function of  interest and productive 
            Args:
                data{pd.DataFrame}: The data use to calculate the loss function weight and it should be training dataset.'''

        self._define_loss_function(data)
        

        self.config.wandb_config['interest_loss_fn']=self.interest_loss_fn
        self.config.wandb_config['productive_loss_fn']=self.productive_loss_fn


        logging.info('loss function loaded...')




    def load_data(self, batch_size: int = 16) -> tuple[DataLoader,DataLoader,DataLoader]:
        '''load the data

            Returns:
                Three DataLoader objects in order of: train, validation, test set '''
        self.config.wandb_config['batch_size']=batch_size

        data = self.db.get_data(table_name='train_data')
        
        data.loc[:,'duration'] = iso_duration_transform(data.loc[:,'duration'])

        train_set,val_set,test_set = data_split(data,seed=self.config.seed)

        self._set_loss_fn(train_set)


        train_dataloader = self.loader.train_dataloader(train_set, batch_size)

        validation_dataloader = self.loader.dataloader(val_set, batch_size, shuffle = False)

        test_dataloader = self.loader.dataloader(test_set, batch_size, shuffle = False)
        

        return train_dataloader, validation_dataloader, test_dataloader




    def _calc_model_loss(self,outputs:dict,batch:BatchEncoding,batch_idx:int, if_wandb:bool=True) ->torch.Tensor :
        '''calculate the model head loss and return the total loss'''

        pred_productive = outputs['productive_rate']
        pred_interest=outputs['interest']

        interest=batch['interest'].to(DEVICE)
        productive=batch['productive_rate'].to(DEVICE)
     

        raw_interest_loss = self.interest_loss_fn(pred_interest,interest)
        raw_productive_loss=self.productive_loss_fn(pred_productive,productive)

        #fillna with 0
        interest_loss=torch.nan_to_num(raw_interest_loss,0.0)
        productive_loss=torch.nan_to_num(raw_productive_loss,0.0)
        
        productive_weight=self.config.productive_loss_weight
        interest_weight=self.config.interest_loss_weight
        total_loss=interest_loss*interest_weight+productive_loss+productive_weight

        if batch_idx%10 == 0:

            logging.info(f'Current batch index: {batch_idx}; productive loss: {productive_loss} ; intereset loss: {interest_loss} ')

        if if_wandb:
            wandb.log({'productive_loss':productive_loss,'interest_loss':interest_loss,'train_loss':total_loss})

        return total_loss


    def _update_train_metrics(self,batch:BatchEncoding,prediction:pd.DataFrame) ->None:
        '''update the train metrics'''

        masked_interest,masked_pred_interest=self._prepare_interest_for_metrics(batch,prediction)
        masked_productive,masked_pred_productive=self._prepare_productive_for_metrics(batch,prediction)

        #Update the mmetrics only if current batch have real data
        if masked_pred_interest.numel()>0: 
            self.interest_train_metrics.update(masked_pred_interest,masked_interest)

        if masked_pred_productive.numel()>0:
            self.productive_train_metrics.update(masked_pred_productive,masked_productive)

    def train_model(self,data:DataLoader, if_wandb:bool=True) ->None:
        '''the training loop of the productive model'''

        self.model.train()

        full_batch_loss=0
        batch_num_count=0

        for batch_idx,batch in enumerate(data):

            outputs=self.model.predict_step(batch)

            self._update_train_metrics(batch,outputs)

            total_loss=self._calc_model_loss(outputs,batch,batch_idx=batch_idx,if_wandb=if_wandb)

            total_loss.backward()


            self.optimizer.step()
                
            self.optimizer.zero_grad()

            full_batch_loss+=total_loss.item()
            batch_num_count+=1

        average_full_batch_loss=full_batch_loss/batch_num_count
        logging.info(f'aver full batch loss: {average_full_batch_loss}')

    

    def _calculate_eval_metrics(self,all_outputs:dict) ->tuple[float,float]:
        '''log the accuracy and f1 scores of the model'''

        interest_accuracy=accuracy_score(all_outputs['actual_interest'],all_outputs['preds_interest'])

        interest_f1=f1_score(all_outputs['actual_interest'],all_outputs['preds_interest'])

        productive_accuracy=accuracy_score(all_outputs['actual_productive'],all_outputs['preds_productive'])

        productive_f1=f1_score(all_outputs['actual_productive'],all_outputs['preds_productive'])

        
        logging.info(f'interest accuracy: {interest_accuracy}')
        logging.info(f'interest_f1: {interest_f1}')

        logging.info('')
        logging.info(f' productive accuracy: {productive_accuracy} ')
        logging.info(f'productive f1: {productive_f1} ')

        return interest_f1,productive_f1



    def _get_model_output(self,logit_value:torch.Tensor) -> torch.Tensor :
        '''get the argmaxed output'''


        pred_value=torch.argmax(logit_value,dim=1) #argmax return highest value's index
    
        assert isinstance(pred_value,torch.Tensor)

        return pred_value

    def _prepare_productive_for_metrics(self,batch:BatchEncoding,prediction:dict) -> tuple[torch.Tensor,torch.Tensor]:
        '''prepare interest data for metrics, return in order of masked_productive,masked_pred_productive'''


        productive_rate=batch['productive_rate']
        productive_ignore_mask= productive_rate!=self.config.ignore_index #only has 1 dimension
        masked_productive=productive_rate[productive_ignore_mask].to(DEVICE)



        logit_productive_rate=prediction['productive_rate']
        pred_productive=self._get_model_output(logit_productive_rate)
        masked_pred_productive=pred_productive[productive_ignore_mask].to(DEVICE)


        return masked_productive,masked_pred_productive
    

    def _prepare_interest_for_metrics(self,batch:BatchEncoding,prediction:dict) -> tuple[torch.Tensor,torch.Tensor]:
        '''prepare interest data for metrics, return in order of masked_interest,masked_pred_interest'''


        interest=batch['interest']
        interest_ignore_mask= interest!=self.config.ignore_index #only has 1 dimension
        masked_interest=interest[interest_ignore_mask].to(DEVICE)

        logit_interest=prediction['interest']
        pred_interest=self._get_model_output(logit_interest)
        masked_pred_interest=pred_interest[interest_ignore_mask].to(DEVICE)


        return masked_interest,masked_pred_interest

    def _update_val_metrics(self,batch:BatchEncoding,prediction:torch.Tensor) ->None:
        '''Update the validation metrics'''
        masked_interest,masked_pred_interest=self._prepare_interest_for_metrics(batch,prediction)
        masked_productive,masked_pred_productive=self._prepare_productive_for_metrics(batch,prediction)

        #Update the mmetrics only if current batch have real data
        if masked_pred_interest.numel()>0: 
            self.interest_val_metrics.update(masked_pred_interest,masked_interest)

        if masked_pred_productive.numel()>0:
            self.productive_val_metrics.update(masked_pred_productive,masked_productive)


    def evaluate_model(self,validation_data:DataLoader,load_model:str|None=None) -> None:
        '''evaluate the model'''

        if load_model is not None:

            self.model=if_load_model(self.model,load_model,lora=self.model.config.use_lora)
            logging.info('load model...')

            
        self.model.eval()


        with torch.no_grad():
            for batch in validation_data:

                prediction=self.model.predict_step(batch)

                self._update_val_metrics(batch,prediction)
                
                


    def _save_highest_ema_model(self,curr_ema:float,best_ema:float,model_pt:str|None) ->None:
        '''Compare current EMA f1 and the best EMA f1 , if current one is higher than the best, override/save the model'''
        if model_pt:#If provided model name

            if curr_ema > best_ema:

                self.save_model(self.model.state_dict(),model_pt)

                best_ema=curr_ema

                logging.info('best_ema_f1 updated | model saved')


    def _calc_curr_ema_f1(self,epoch_num:int,interest_f1,productive_f1) ->float:
        '''Calculate the current EMA f1'''
        alpha=self.config.ema_alpha
        
        productive_weight=self.config.ema_productive_weight
        interest_weight=1-productive_weight


        weighted_f1=interest_weight*interest_f1+productive_f1*productive_weight

        if epoch_num==0:
            curr_ema_f1=weighted_f1

        else:
            curr_ema_f1 =(alpha * weighted_f1) + (1-alpha) *weighted_f1

        return curr_ema_f1

    def _compute_metrics(self,epoch:int) ->tuple[float,float]:
        '''Compute and log the train val metrics to wandb'''

        interest_train=self.interest_train_metrics.compute()
        productive_train=self.productive_train_metrics.compute()

        interest_val=self.interest_val_metrics.compute()
        productive_val=self.productive_val_metrics.compute()
        logging.info('Interest validation f1')
        logging.info(interest_val)
        logging.info('productive validation f1')
        logging.info(productive_val)

        wandb.log({
            'epoch':epoch,
            'interest_train_f1':interest_train['f1'],
            'interest_train_precision':interest_train['precision'],
            'interest_train_recall':interest_train['recall'],

            'productive_train_f1':productive_train['f1'],
            'productive_train_precision':productive_train['precision'],
            'productive_train_recall':productive_train['recall'],

            'interest_val_f1':interest_val['f1'],
            'interest_val_precision':interest_val['precision'],
            'interest_val_recall':interest_val['recall'],

            'productive_val_f1':productive_val['f1'],
            'productive_val_precision':productive_val['precision'],
            'productive_val_recall':productive_val['recall'],
        })

        self.interest_train_metrics.reset()
        self.productive_train_metrics.reset()
        self.interest_val_metrics.reset()
        self.productive_val_metrics.reset()

        return interest_val['f1'],productive_val['f1']



    def epoch_training_loop(self,total_epoch: int, train_data: DataLoader, val_data: DataLoader, model_pt: str|None=None) -> dict:
        '''train the model in epoch loop
        Args:
            total_epoch: the number of epoch you want to train,
            train_data: the training dataset,
            val_data: the validation dataset,
            model_pt: the model save path, in default it's None, which mean it won't save the model
            
            
        Return:
            A dictionary that contain the f1 scores of every epoch of every head (i.e. interest and productive_rate)    
            '''

        f1_dict={
            'interest':[],
            'productive_rate':[]
        }
        best_ema_f1=0.
        curr_ema_f1=0.


        for epoch in range(total_epoch):
            logging.info(f'current epoch: {epoch+1}/ {total_epoch}')
            self.train_model(train_data)

            self.evaluate_model(val_data)
            
            interest_val_f1,productive_val_f1=self._compute_metrics(epoch)

            curr_ema_f1=self._calc_curr_ema_f1(epoch,interest_val_f1,productive_val_f1)

            self._save_highest_ema_model(curr_ema_f1,best_ema_f1,model_pt)



            f1_dict['interest'].append(interest_val_f1)
            f1_dict['productive_rate'].append(productive_val_f1)
            logging.info(f'current_ema_f1: {curr_ema_f1} | best_ema_f1 : {best_ema_f1}')



        return f1_dict

    def _check_if_name_exists(self,name:str) ->bool:
        '''check if current model name is already in the artifacts folder'''
        logging.info(f'target path {ARTIFACTS_PATH/name}')

        if os.path.isfile(ARTIFACTS_PATH/name):

            answer=input(f'The file already existed, are you sure you wanna overwrite {name} and continue training? y/n: ')

            if answer =='y':
                    
                logging.info("Training process continue!")

                return True

            elif answer =='n':
                logging.info('training process stopped!')

                return False
            else:
                logging.info('invalid answer, process stopped!')

                return False

        return True
        


    def start_train(
            self,
            model_name:str='new_model',
            run_name:str|None=None,
            check_file_exist:bool=True
            ) ->None :
        '''
        start training with epoch training loop
        
        Args:
            model_name (str): The name of artifacts file. Default to `new_model`
            run_name(str): The run name in Wandb. Default to model name
            check_file_exist(bool): Whether check artifacts file name already exists. Default to enable.
        
        '''
        
        self._set_seed()

        train_set,val_set,test_set=self.load_data(batch_size=self.config.batch_size)

        #set the run name to model name if there's no name provided
        if run_name is None:
            run_name=model_name


        if check_file_exist:
            print('checking file existence')
            if_train=self._check_if_name_exists(name=model_name)
        else:
            if_train=True

        if if_train:

            wandb.init(
            project='personal_feed',
            name=run_name,
            config=self.config.wandb_config)

            wandb.watch(self.model,log='all',log_freq=10)

            self.epoch_training_loop(total_epoch=self.config.total_epoch,train_data=train_set,val_data=val_set,model_pt=model_name)

            wandb.finish()



    def _set_seed(self,seed:int = 42) -> None:
        '''set the random seed for random,numpy,torch mps,os,sklearn and torch generator'''

        set_random_seed(seed)

        self.config.seed=seed
        self.config.g=torch.Generator().manual_seed(seed)


    def _save_model_dict(self,state_dict:dict,model_path:str) ->None:
        '''detect if the state_dict us model or not, and choose save method base on that'''

        if not self.model.config.use_lora:
            torch.save(state_dict,ARTIFACTS_PATH/model_path)

        elif self.model.config.use_lora:
            self._save_lora(state_dict,model_path)
            
                    
        else:
            logging.error(f'Error: the model_or_dict input is either Module nor state dict, unknown type:{type(state_dict)}')

    def _save_lora(self,state_dict:dict,lora_path:str) -> None:
        '''save only lora adapters'''

        lora_dict={}

        for name,param in state_dict.items():
            if 'lora_A' in name or 'lora_B' in name or 'prompt_tuning' in name:
                lora_dict[name] = param

        torch.save(lora_dict,ARTIFACTS_PATH/lora_path)



    def save_model(self,model_or_state_dict:Module|dict,model_name:str,wandb_save:bool =True) -> None:
        '''save the target model
        
        Args:
            model_or_state_dict (Module| dict ): THe target model itself or the state dict from the target model
            model_name (str): the path you want to save the model
            wandb_save (bool): Whether save the artifact to wanndb
            '''

        model_name=f'{model_name}'

        if wandb_save:
            model_artifact=wandb.Artifact(
                f'{model_name}',
                type='model',
                description=f'{model_name}'
            )
        
        if isinstance(model_or_state_dict,Module):
            model_dict=model_or_state_dict.state_dict()
        else:
            model_dict=model_or_state_dict
        

        self._save_model_dict(model_dict,model_name)
        logging.info(f"{model_name} model saved!")

        

        if wandb_save:
            model_artifact.add_file(ARTIFACTS_PATH/model_name)
            wandb.log_artifact(model_artifact)



class HybridProductiveModelTraining(ProductiveModelTraining[HybridProductiveModelConfig]):
    '''add aw data input and the sentence transformer as its encoder,
        the aw data vector would bypass the main model's embeding layer'''

    def __init__(self,config:HybridProductiveModelConfig|None = None):

        self.config=self._setup_config(config)

        model= HybridProductiveModel(config=self.config).to(DEVICE)


        super().__init__(model=model,config=self.config)


        self.scaler=GradScaler()


        self.loader=HybridProductiveLoader(self.config)

    def _setup_config(self,config:HybridProductiveModelConfig|None):
        '''Initial training config'''

        if config is None:
            logging.info('HybridProductiveModel is using default config')
            return HybridProductiveModelConfig()
            

        return config


    
    def _define_loss_function(self,data:pd.DataFrame) ->None:
        '''define the two head's loss function'''

        interest_weight,productive_weight=self._calc_head_loss(data)

        
        self.interest_loss_fn=CrossEntropyLoss(
            weight=interest_weight,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.interest_label_smooth
            )

        self.productive_loss_fn=CrossEntropyLoss(
            weight=productive_weight,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.productive_label_smooth
            )
    
    

    def train_model(self,data:DataLoader, if_wandb:bool=True) ->None:
        '''the training loop of the hybrid productive model'''

        self.model.train()

        accumulation_steps=self.config.accumulation_steps

        full_batch_loss = 0
        batch_num_count = 0

        for i,batch in enumerate(data):
            
            with autocast(device_type=DEVICE.type):#torch.device.type is the string name of the device(e.g. 'cuda')
            
                outputs=self.model.predict_step(batch)

                total_loss=self._calc_model_loss(outputs,batch,i,if_wandb=if_wandb)

                total_loss=total_loss/accumulation_steps
            
            self._update_train_metrics(batch,outputs)
            
            
            self.scaler.scale(total_loss).backward()

            if (i+1)%accumulation_steps==0:

                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.)

                self.scaler.step(self.optimizer)
                
                self.scaler.update()
                self.optimizer.zero_grad()


            full_batch_loss+=total_loss.item()
            batch_num_count=i

        logging.info(f'aver full batch loss: {full_batch_loss/batch_num_count}')

    
    def _clac_f1_mean_and_std(self,f1_dict:dict) ->None:
        '''calculate the mean and standard deviation of f1 score in every epoch'''

        interest_f1_mean=np.mean(f1_dict['interest'])
        interest_f1_std=np.std(f1_dict['interest'])


        productive_f1_mean=np.mean(f1_dict['productive_rate'])
        productive_f1_std=np.std(f1_dict['productive_rate'])

        
        logging.info(f'interest f1 mean: {interest_f1_mean} ')
        logging.info(f'interest f1 std: {interest_f1_std}')
        logging.info('')

        logging.info(f'productive f1 mean: {productive_f1_mean}')
        logging.info(f'productive f1 std:{productive_f1_std}')
       



    def kfold_train(self,group_name):
        '''train the model by using K Fold to test model performance '''


        dataset=self.db.get_data('train_data')
        dataset.loc[:,'duration'] = iso_duration_transform(dataset.loc[:,'duration'])

        kfold=KFold(n_splits=5,shuffle=True,random_state=self.config.seed)


        for fold,(train_id,val_id) in enumerate(kfold.split(dataset)):
            logging.info(f'Fold {fold+1} / 5')
    
            wandb.init(
                project='personal_feed',
                config=self.config.wandb_config,
                name=f'{group_name}_{fold}',
                reinit=True,
                group=group_name,
                job_type='kfold_train')

            wandb.watch(self.model,log='all',log_freq=10)
            
            self.model=HybridProductiveModel(self.config).to(DEVICE)

            self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=5e-5)#reset the optimizer


            train_set=dataset.loc[train_id.flatten()]

            val_set=dataset.loc[val_id.flatten(),:]

            self._set_loss_fn(train_set)

            train_loader=self.loader.train_dataloader(train_set,batch_size=4)

            val_loader=self.loader.dataloader(val_set,batch_size=self.config.batch_size,shuffle=False)

            f1_dict=self.epoch_training_loop(total_epoch=self.config.total_epoch,train_data=train_loader,val_data=val_loader)
            

            self._clac_f1_mean_and_std(f1_dict)

            wandb.finish()
        
        

    def kfold_start(self,group_name:str='default_group') ->None:
        '''start the kfold training '''

        self._set_seed()


        self.kfold_train(group_name=group_name)








            






    







    

