

import os
import requests
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from src.inference.productive import HybirdProductiveModelPredicting
from src.utils.ops import like_dislike_data_preprocess,inerest_productive_data_preprocess,timestamp_data_preprocess
from Personal_feed.src.pipelines.productive import ProductiveModelTraining,HybirdProductiveModelTraining
from Personal_feed.src.models.productive import HybirdProductiveModel
from Personal_feed.src.web.website_frontend import new_feed_arrive
from Personal_feed.src.db.database import PersonalFeedDatabase

from config import HybirdProductiveModelConfig


def feed_update() ->None:
    '''use model to predict feed and update the feed in the website'''
    model=HybirdProductiveModelPredicting('hybird_productive_model_4BS_5E_SwiGLU_AdaLN.pth')
    model.get_preds_from_hybird_productive_model(time_range=300)
    

    
    requests.post('http://0.0.0.0:8080/trigger-update')
    print('website_notified')

  

def run_scheduler():
    '''run the scheduler to update website feed from 8 AM to 23 PM and 0AM to 4AM per 30 minutes'''
    scheduler=BlockingScheduler()
    trigger=CronTrigger(hour='8-23,0-4',minute='*/30')

    scheduler.add_job(
        func=feed_update,
        trigger=trigger,
        next_run_time=datetime.now()
    )
    scheduler.start()

if __name__=='__main__':
    run_scheduler()


'''


if __name__=='__main__':
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

    
    test=HybirdProductiveModelTraining()

    test.start_train(model_name='hybird_productive_model_4BS_5E_SwiGLU_AdaLN.pth',total_epoch=5,batch_size=2)
    





    model_name = 'MT_like_dislike_data_retrain_tagintereset_1_interestCEL_clip_grad_norm.pth'
    
    compare_model_name='MT_like_dislike_data_retrain_tagintereset_1_pre_tag_fix.pth'
    model = MultiTaskModel().to(device='mps')

    training_process = Training(model)
    training_process.set_seed(42)

    
    train_data,val_data,test_data = training_process.load_data(16)

    training_process.epoch_training_loop(6,train_data,val_data,model_name)

    #training_process.evaluate_model(test_data)

    '''
    
