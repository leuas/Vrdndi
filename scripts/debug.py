'''the file for debug '''
import pprint
import pandas as pd
import os
import sqlite3 as sq
import torch
import numpy as np
from src.config import PROJECT_ROOT,HOST,HOSTNAME,PORT,TRAIN_DATA_PATH
from src.db.database import PersonalFeedDatabase

from src.utils.data_etl import get_and_save_his_data_for_database,get_and_save_liked_disliked_data_for_database

from src.utils.ops import inerest_productive_data_preprocess

from src.inference.productive import HybirdProductiveModelPredicting


if __name__=='__main__':
    
    

    
    
    
   

    


    



    




'''

 with sq.connect(db.dbpath) as conn:
                query="CREATE TABLE IF NOT EXISTS feed_state( current_state INTEGER DEFAULT 0)"

                conn.execute(query)

                cursor=conn.execute("SELECT count(*) FROM feed_state")

                if cursor.fetchone()[0] == 0:
                    cursor.execute("INSERT INTO feed_state (current_state) VALUES (0)")

test=HybirdProductiveModelPredicting(model_name='hybird_productive_model_4BS_inner_projector_GA_10_epoch.pth')

pprint.pprint(test.get_preds_from_hybird_productive_model())



model=MultiTaskModel().to('mps')

test=Training(model)

train_set,val_set,test_set=test.load_data(file_name="like_dislike_tag_data_tag's_interest=1.csv")

test.epoch_training_loop(total_epoch=5,train_data=train_set,val_data=val_set)






def re(it):
    def de(fc):

        return fc(it)
        
    return de

@re(1000)
def my_funct(n):
    return 2*n

my_funct(3)

'''



