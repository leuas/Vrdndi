"""
Basic Inference Demo for hybrid productive model

This script shows the functionality of the hybrid productive model by performing 
inference on video data which is from `test/fixture` and a fake short app sequence
from AW local server. It is primarily used as a really quick start.

Modes:
    - Standard: Loads the LoRA layers from `artifacts/` and runs the inference.
    - Dry Run:  Run the inference directly.

Purpose:
    - Verifies **some** library dependencies (e.g. torch, transformers).
    - DOES NOT validate Youtube API, Database, Website, main training pipelines, etc


This demo has a secondary function, `predict_improperly`, to simulate failure 
handling. It shows the cases where the offline encoded tensor is unavailable, 
triggering a warning for each missing file (in this case, all files) and replacing 
the missing data with a zero tensor as fallback.

Usage:
    python demo.py

Notes:
    By default, this script runs `predict_normally()` and `mode='standard'`. To test the fallback 
    mechanism or a different mode, open this file and switch the function call to `predict_improperly()` or switch the mode.

Expected Output:
    - `predict_normally()`:
    Initialization logs followed by the final output DataFrame.

    - `predict_improperly`:
        Extensive warning logs regarding missing offline tensors, followed by the 
        final output DataFrame .
        
"""

import pandas as pd
from unittest.mock import patch
from datetime import datetime

from src.inference.productive import HybridProductiveModelPredicting
from src.path import FIXTURE_PATH

class Demo:
    '''Initializes the demo for model functionality showcase
        
        Args:
            mode: 'standard' loads the real LoRA weights from fixtures.
                  'dry_run' skips loading to test pipeline logic only.
    '''

    def __init__(self,mode:Literal['standard','dry-run']='standard') -> None:
        
        if mode == 'standard':
            self.model_inference=HybridProductiveModelPredicting('example_model.pth')
        else:
            self.model_inference=HybridProductiveModelPredicting()
        
        self.drop_columns=['interest','tag']


    def predict_normally(self) ->None:
        '''Predict fake video data with offline encoded app sequence'''

        with patch('src.utils.data_etl.ActivityWatchClient.query') as mock_aw_sequence:
            
            mock_aw_sequence.return_value=fake_aw_sequence
            
            video_data=pd.read_csv(FIXTURE_PATH/'fake_video_data.csv').drop(columns=self.drop_columns)


            inference_data=self.model_inference.prepare_predicting_data(inference_data=video_data)

            self.model_inference.predict(inference_data=inference_data,update_db=False)

    def predict_improperly(self) ->None:
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
    







        