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
    '''A demo for model funtional showcase'''

    def __init__(self) -> None:
        
        self.model_inference=HybirdProductiveModelPredicting()
        
        self.drop_columns=['interest','tag']


    def predict_normally(self) ->None:
        '''Predict fake vide data with offline encoded app sequence'''

        with patch('src.utils.data_etl.ActivityWatchClient.query') as mock_aw_sequence:
            
            mock_aw_sequence.return_value=fake_aw_sequence
            
            video_data=pd.read_csv(FIXTURE_PATH/'fake_video_data.csv').drop(columns=self.drop_columns)


            inference_data=self.model_inference.prepare_predicting_data(inference_data=video_data)

            self.model_inference.predict(inference_data=inference_data,update_db=False)

    def predict_unproperly(self) ->None:
        '''Predict fake video data without offline encoded app sequence '''

        with patch('src.utils.data_etl.ActivityWatchClient.query') as mock_aw_sequence:
                
            mock_aw_sequence.return_value=fake_aw_sequence

            video_data=pd.read_csv(FIXTURE_PATH/'fake_video_data.csv').drop(columns=self.drop_columns)
            video_data['timestamp']=datetime.now().isoformat()
            self.model_inference.predict(inference_data=video_data,update_db=False)

#The fake app seuqence data that is fetched from Activity Watcher
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
    demo=Demo()

    demo.predict_normally()
    







        