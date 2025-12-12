'''the config file of the project'''
import torch

from pathlib import Path

from dataclasses import dataclass
from typing import TypeVar


#path constant

current_file_path=Path(__file__)
PROJECT_ROOT=Path(__file__).parent.parent



TEST_PATH=PROJECT_ROOT/'test'

FIXTURE_PATH=TEST_PATH/'fixture'

AW_TENSOR_PATH=PROJECT_ROOT/'aw_tensor'





SRC=PROJECT_ROOT/'src'


ASSETS_PATH=SRC/'assets'


LEGACY_PATH=PROJECT_ROOT/'legacy'

DEVICE=torch.device('mps' if torch.mps.is_available() else 'cuda' )

HOSTNAME='leuasMacBook-Air.local'
HOST='100.100.6.64'
PORT=5600


@dataclass
class ProductiveModelConfig:
    ''' THe config of Productive model'''
    model_name:str ='BAAI/bge-m3'
    lr=5e-5
    weight_decay=1e-3
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
class HybirdProductiveModelConfig(ProductiveModelConfig):
    '''The config of HybirdProductiveModel '''


    num_in_feature:int =3 #duration,time_sin,time_cos
    num_out_feature:int =384

    cond_dim:int=1 #duration


    train_num_workers:int=0
    eval_test_num_workers:int=0

    max_length:int=8094

    sampler_interest_ratio:float=0.5
    sampler_productive_ratio:float=1-sampler_interest_ratio

    
    accumulation_steps:int=4

    interest_label_smooth:float=0
    productive_label_smooth:float=0







    


