'''Contain the function that used productive model to make prediction'''
import torch
import pprint
import pandas as pd

from datetime  import datetime
from src.db.database import VrdndiDatabase
from src.pipelines.productive import HybirdProductiveModelTraining
from src.models.productive import HybirdProductiveModel

from torch.utils.data import DataLoader

from src.utils.ops import duration_transform,text_col_fillna,convert_timestamp_to_pt_file,if_load_model,iso_duration_transform
from src.model_dataset.loader import HybirdProductiveLoader


from src.config import DEVICE,HybirdProductiveModelConfig
from src.path import INFERENCE_DATA_PATH


class HybirdProductiveModelPredicting:
    '''the class that contain the process of using HyvirdProductiveModel to predict value'''
    def __init__(self,model_name:str|None =None,config:HybirdProductiveModelConfig|None=None) -> None:

        if config is None:
            self.config = HybirdProductiveModelConfig()
            print('HybirdProductiveModel is using default config')

        else:
            self.config=config

        if model_name is None:
            self.model= HybirdProductiveModel(self.config).to(DEVICE)
        else:
            self.model= HybirdProductiveModel(self.config).to(DEVICE)
            self.model=torch.compile(self.model,mode='reduce-overhead')
            self.model = if_load_model(self.model,model_name,lora=self.model.config.use_lora)

        
        self.loader = HybirdProductiveLoader(self.config)

        self.db=VrdndiDatabase()


    def _load_feed_data(self,data:pd.DataFrame,batch_size:int = 32) ->DataLoader:
        '''load the feed data'''
        
        data.loc[:,'duration'] = iso_duration_transform(data['duration'])

        data=text_col_fillna(data)

        feed_dataloader=self.loader.dataloader(data,batch_size=batch_size,path=INFERENCE_DATA_PATH)

        return feed_dataloader
    

    def _prepare_predicting_data(self,time:datetime|None = None,time_range:int=7) ->pd.DataFrame:
        '''prepare the aw seuqence data for predicting
            
            Args:
                time{datetime}: The time mode used to predict what feed user(you) should watch at that time
                time_range{int}: The lookback windows for video retrival.
                    Filter the database for videos uploaded within these many days prior to time argment.
                     
            '''
        if time is None:

            now=datetime.now().astimezone()
        else :
            now = time


        data=self.db.fetch_videos_from_past_days(time_range)

        data['timestamp']=now.isoformat()

        convert_timestamp_to_pt_file(now,INFERENCE_DATA_PATH)


        return data



    def get_preds_from_hybird_productive_model(self,time:datetime|None = None,time_range:int=7) ->None:
        ''' get the prediction from productive model'''
        
        data=self._prepare_predicting_data(time,time_range)
        print(data)

        dataloader=self._load_feed_data(data=data)

        outputs={
            'productive_rate':[],
            'interest':[],
            'videoId':[]
        }
        self.model.eval()

        with torch.no_grad():
            
            for batch in dataloader:

                prediction=self.model.predict_step(batch)

                interest=prediction['interest'].cpu().numpy()

                productive_rate=prediction['productive_rate'].cpu().numpy()

                interest_s=pd.Series(interest[:,1].tolist())
                productive_s=pd.Series(productive_rate[:,1].tolist())

                outputs['interest'].append(interest_s)#There's two columns, one for uninterest, one for inter
                outputs['productive_rate'].append(productive_s) #As above
                outputs['videoId'].append(batch['videoId_series'])

                print(batch['videoId_series'])

                

                
        interest_series=pd.concat(outputs['interest'])
        productive_series=pd.concat(outputs['productive_rate'])

        videoid_series=pd.concat(outputs['videoId'])

        predicts_data=pd.concat([videoid_series,interest_series,productive_series],ignore_index=True,axis=1)
        
        predicts_data=predicts_data.rename(columns={0:'videoId',1:'interest',2:'productive_rate'})
            
        pprint.pprint(predicts_data)

        contain_predic_data=data.merge(
            predicts_data,
            how='left',
            left_on='videoId',
            right_on='videoId'
        )

        self.db.update_feed(contain_predic_data)


            
 
        
    

    





    






