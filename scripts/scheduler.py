'''the script run constantly to update website feed'''
import logging
import os
import requests
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from src.inference.productive import HybridProductiveModelPredicting
from src.config import HybridProductiveModelConfig



def feed_update() ->None:
    '''use model to predict feed and update the feed in the website'''
    
    config=HybridProductiveModelConfig()
    config.eval_test_num_workers=4
    model=HybridProductiveModelPredicting(config=config)
    data=model.prepare_predicting_data(time_range=300)
    model.predict(inference_data=data)

    #experimental feature, if you didn't enable, then it does nothing.
    requests.post('http://127.0.0.1:8080/trigger-update')


  

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