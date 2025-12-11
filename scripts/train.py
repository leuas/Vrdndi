'''the script run the training logic of productive model'''

import os

from src.utils.ops import like_dislike_data_preprocess,inerest_productive_data_preprocess,timestamp_data_preprocess
from src.pipelines.productive import HybirdProductiveModelTraining
from src.config import HybirdProductiveModelConfig,TRAIN_DATA_PATH


if __name__=='__main__':
    os.environ['no_proxy'] = '100.100.6.64'


    # Set a main "cache" folder
    main_cache_dir = 'D:\\hg-cache'
    temp_dir = 'D:\\Temp' # The temp folder you created

    # --- Python & System Temp ---
    # Force Python's tempfile module to use D:\Temp
    os.environ['TEMP'] = temp_dir
    os.environ['TMP'] = temp_dir
    os.environ['TMPDIR'] = temp_dir # For non-windows systems, good to have

    # --- Hugging Face Cache ---
    os.environ['HF_HOME'] = os.path.join(main_cache_dir, 'hub')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(main_cache_dir, 'datasets')

    # --- Wandb (The Most Important Part) ---
    # Tell wandb to put *everything* in a subfolder on D:
    wandb_dir = os.path.join(main_cache_dir, 'wandb')
    os.environ['WANDB_DIR'] = wandb_dir
    os.environ['WANDB_CACHE_DIR'] = os.path.join(wandb_dir, 'cache')
    os.environ['WANDB_CONFIG_DIR'] = os.path.join(wandb_dir, 'config')
    os.environ['WANDB_DATA_DIR'] = os.path.join(wandb_dir, 'data') # This should control artifact staging


    config=HybirdProductiveModelConfig()

    config.train_num_workers=4
    config.eval_test_num_workers=4
    config.accumulation_steps=4

    config.interest_loss_weight=0.33

    config.sampler_interest_ratio=3/4
    config.productive_output_layer_dropout=0.5
    
    test1=HybirdProductiveModelTraining(config=config)


    #test1.start_train(model_name='hybird_productive_model_4BS_10E_ratio31_equal_400D_drop_noisy.pth',total_epoch=10,batch_size=4)
    
    
    test1.kfold_start()
    