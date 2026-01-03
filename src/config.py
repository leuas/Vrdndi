'''the config file of the project'''
import os
import logging
import torch

from dotenv import load_dotenv

from dataclasses import dataclass
from typing import TypeVar

DEVICE=torch.device('mps' if torch.mps.is_available() else 'cuda' )


load_dotenv()

CLIENT_SECRET_FILE=os.getenv('CLIENT_SECRET_FILE','')

HOSTNAME=os.getenv('HOSTNAME',None)
HOST=os.getenv('HOST',None)
PORT=os.getenv('PORT',None)

WEBSITE_HOST='0.0.0.0'

ENABLE_EXPERIMENTAL_FEATURES = False #Enable some feature that is still debuging and testing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
    )



@dataclass
class ProductiveModelConfig:
    ''' THe config of Productive model'''
    model_name:str ='BAAI/bge-m3'
    lr=5e-5
    weight_decay=0.01
    compile_model:bool=True

    productive_out_feature:int=2
    interest_out_feature:int=2

    productive_output_layer_dropout:float=0.1
    interest_output_layer_dropout:float=0.1

    use_lora:bool = True
    lora_rank:int = 8
    lora_alpha:int =16
    lora_target_modules:str ='all-linear' 

    ignore_index:int=-100

    productive_loss_weight=1
    interest_loss_weight=1

    ema_alpha:float=0.6
    ema_productive_weight:float=0.65

    seed=42 #defualt if you didn't set the random seed
    g=None

    wandb_config={
        'learning_rate':lr,
        }
    
    batch_size:int=4
    total_epoch:int=10
    
    

@dataclass
class HybridProductiveModelConfig(ProductiveModelConfig):
    '''The config of HybridProductiveModel '''


    num_in_feature:int =3 #duration,time_sin,time_cos
    num_out_feature:int =384

    cond_dim:int=1 #duration


    train_num_workers:int=0
    eval_test_num_workers:int=0

    max_length:int=8192

    sampler_interest_ratio:float=0.5
    sampler_productive_ratio:float=1-sampler_interest_ratio

    
    accumulation_steps:int=4

    interest_label_smooth:float=0
    productive_label_smooth:float=0




@dataclass
class RecursiveBGEConfig:
    '''Config of recursive BGE-M3 model'''
    ori_model_name:str="BAAI/bge-m3"
    batch_size:int=4
    max_steps:int=36
    tau:float=0.01
    lr:float=1e-4
    total_epoch:int=5

    init_layer_index:int=0

    buffer_size:int=10000
    seed:int=42

    num_workers:int=0

    wandb_config={}

    train_batch_num:int=1000

    eval_batch_num:int=200

    debug_mode:bool=False

    max_lengh:int=8192

    accumulation_steps:int=16
    

    


