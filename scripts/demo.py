'''Model demo'''
import os
import pprint
import pandas as pd

from datetime import datetime,timedelta

from src.utils.ops import productive_data_preprocess,convert_timestamp_to_pt_file

from src.db.database import VrdndiDatabase
from src.inference.productive import HybirdProductiveModelPredicting
from src.path import INFERENCE_DATA_PATH

class Demo:
    '''A demo for model performance showcase'''

    def __init__(self) -> None:
        
        self.db=VrdndiDatabase()

        self.inference=HybirdProductiveModelPredicting()
    def get_unique_items(self,input_df:pd.DataFrame) ->None:
        '''Find the datapoint that belong to same itemds but has different feedback (in different time)
        '''


        unique_counts=input_df.groupby('videoId')['productive_rate'].transform('nunique')

        mask=unique_counts>1

        pprint.pprint(input_df[mask])

    def predict_feedback(self):
        '''use model to predict feedback in different time'''

        now=datetime.now()

        print(now)

        to_hours=timedelta(hours=12)

        last_night=now-to_hours
        print(last_night)

        time_list=[now,last_night]

        
        data=productive_data_preprocess()

        data_list=[]

        for time in time_list:
            convert_timestamp_to_pt_file(time,path=INFERENCE_DATA_PATH)
            data['timestamp']=time

            prediction=self.inference.predict(time=time,inference_data=data)

            data_list.append(prediction)


        cat_df=pd.concat(data_list,ignore_index=True)

        self.get_unique_items(cat_df)


if __name__=='__main__':
    os.environ['NO_PROXY']='100.100.6.64'
    demo=Demo()

    demo.predict_feedback()




        