'''this file contain the prediction function of baseline model'''

import pickle
import pprint
import numpy as np
import pandas as pd


from src.utils.data_etl import prepare_log_model_feed,prepare_rf_model_feed
from src.path import ARTIFACTS_PATH

def make_log_model_prediction() ->pd.DataFrame:
    '''logistic model make prediction'''

    x,raw_feed=prepare_log_model_feed()

    with open(ARTIFACTS_PATH/'logsitic_regression.pkl','rb') as file:
        log_model=pickle.load(file)

    log_y=pd.DataFrame(log_model.predict_proba(x))


    x['interesting']=log_y.loc[:,1]

    interesting_mask=x['interesting']>0.75
    
    video_feed=raw_feed[interesting_mask]
    
    
    return video_feed




def make_rf_prediction() ->pd.DataFrame:
    '''make prediction for category by using random forest'''

    interesting_video=make_log_model_prediction()

    with open(ARTIFACTS_PATH/'randomforest.pkl','rb') as file:
        rf_model=pickle.load(file)

    x=prepare_rf_model_feed(interesting_video)

    y=rf_model.predict(x)

    #keep the video id
    interesting_video['category']=y
    
    return interesting_video

if __name__=='__main__':
    pprint.pprint(make_log_model_prediction())