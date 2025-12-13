'''Model demo'''
import os
import pprint
import pandas as pd
from unittest.mock import patch
from datetime import datetime,timedelta

from src.utils.ops import productive_data_preprocess,convert_timestamp_to_pt_file

from src.db.database import VrdndiDatabase
from src.inference.productive import HybirdProductiveModelPredicting
from src.path import INFERENCE_DATA_PATH,PROCESSED_DATA_PATH,FIXTURE_PATH

class Demo:
    '''A demo for model performance showcase'''

    def __init__(self) -> None:
        
        self.db=VrdndiDatabase()

        self.inference=HybirdProductiveModelPredicting('hybird_productive_model_4BS_10E_ratio31_prodc_label_smooth0.1_dropout0.5.pth')
    def get_unique_items(self,input_df:pd.DataFrame) ->None:
        '''Find the datapoint that belong to same itemds but has different feedback (in different time)
        '''


        unique_counts=input_df.groupby('videoId')['productive_rate'].transform('nunique')

        mask=unique_counts>1

        pprint.pprint(input_df[mask])
        input_df[mask].to_csv(PROCESSED_DATA_PATH/'demo_data.csv')

    def predict_feedback(self):
        '''use model to predict feedback in different time'''

        now=datetime.now()

        print(now.isoformat())

        
        data=self.db.get_feed()
        data=data.drop(columns=['interest','productive_rate'])

        data_list=[]
        convert_timestamp_to_pt_file(pd.Series(now.isoformat()),path=INFERENCE_DATA_PATH)
   
        data['timestamp']=now()

        prediction=self.inference.predict(time=now,inference_data=data,update_db=False)

        data_list.append(prediction)


        merged_df=pd.merge(data_list[0],data_list[1],on='videoId',suffixes=('_0','_1'))

        diff=(merged_df['productive_rate_0']-merged_df['productive_rate_1']).abs()

        changes=merged_df[diff>0.01].drop(columns=['duration_1','interest_1','description_1','title_1','youtuber_1','data_state_1','upload_time_1','duration_0','interest_0','data_state_0','upload_time_0','description_0'])



        pprint.pprint(changes)
        changes.to_csv(PROCESSED_DATA_PATH/'changes.csv')


fake_aw_sequence=[[{'data': {'$category': ['Productivity', 'Gemini'],
            'app': 'Zen',
            'title': 'Google Gemini'},
            'duration': 17.991,
            'id': 186156,
            'timestamp': '2025-12-12T14:22:30.983500Z'},
            {'data': {'$category': ['Productivity', 'Remote Work'],
                        'app': 'Parsec',
                        'title': 'Parsec'},
            'duration': 49.135,
            'id': 186157,
            'timestamp': '2025-12-12T14:22:48.974500Z'},
            {'data': {'$category': ['Comms', 'IM'], 'app': 'QQ', 'title': 'QQ'},
            'duration': 0.66,
            'id': 186158,
            'timestamp': '2025-12-12T14:23:38.109500Z'},
            {'data': {'$category': ['Productivity', 'Gemini'],
                        'app': 'Zen',
                        'title': 'Google Gemini'},
            'duration': 82.302,
            'id': 186159,
            'timestamp': '2025-12-12T14:23:38.769500Z'}]]

if __name__=='__main__':
    os.environ['NO_PROXY']='100.100.6.64'
    
    with patch('src.utils.data_etl.ActivityWatchClient.query') as mock_aw_sequence:
        
        mock_aw_sequence.return_value=fake_aw_sequence

        drop_columns=['interest','tag']

        video_data=pd.read_csv(FIXTURE_PATH/'fake_video_data.csv').drop(columns=drop_columns)

        time=datetime.now()


        model_inference=HybirdProductiveModelPredicting()

        inference_data=model_inference.prepare_predicting_data(inference_data=video_data)

        model_inference.predict(inference_data=inference_data,update_db=False)






        