'''the script run constantly to update website feed'''
import os
import requests
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from src.inference.productive import HybirdProductiveModelPredicting
from src.config import HybirdProductiveModelConfig

os.environ['no_proxy'] = '100.100.6.64'

def feed_update() ->None:
    '''use model to predict feed and update the feed in the website'''
    
    config=HybirdProductiveModelConfig()
    config.eval_test_num_workers=8
    model=HybirdProductiveModelPredicting('hybird_productive_model_4BS_10E_EMA_save_loss_weighted.pth',config=config)
    model.get_preds_from_hybird_productive_model(time_range=300)
    

    
    requests.post('http://127.0.0.1:8080/trigger-update')
    print('website_notified')

  

def run_scheduler():
    '''run the scheduler to update website feed from 8 AM to 23 PM and 0AM to 4AM per 30 minutes'''
    scheduler=BlockingScheduler()
    trigger=CronTrigger(hour='8-23,0-4',minute='*/45')

    scheduler.add_job(
        func=feed_update,
        trigger=trigger,
        next_run_time=datetime.now()
    )
    scheduler.start()


if __name__=='__main__':
    run_scheduler()