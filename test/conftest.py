'''contain the fixture of all test file'''
import pytest
import torch
import pandas as pd
import numpy as np
import pprint
from datetime import datetime

from src.pipelines.productive import HybirdProductiveModelTraining

from src.models.productive import ProductiveModel
from src.models.activity_watcher_encoder import ActivityWatcherEncoder

from src.model_dataset.productive import ProductiveData
from src.utils.ops import set_random_seed,duration_transform,iso_duration_transform
from src.utils.ops import prepare_sentence_transformer_input

from src.inference.productive import HybirdProductiveModelPredicting

from src.config import DEVICE,ProductiveModelConfig
from src.path import FIXTURE_PATH




@pytest.fixture(scope='module')
def set_test_random_seed(seed=666)->None:
    '''set the random seed in a file '''
    set_random_seed(seed=seed)




@pytest.fixture
def get_time() ->datetime:
    '''get a fixed time'''

    return datetime.fromisoformat('2025-11-01T09:31:05.401000+00:00')

@pytest.fixture
def get_hybird_productive_model_training_class() ->HybirdProductiveModelTraining:
    '''get a fresh HybirdProductiveModelTraining class'''

    return HybirdProductiveModelTraining()



@pytest.fixture
def get_aw_data() ->pd.DataFrame:
    '''get the aw data '''

    data=[{'data.$category': ['Uncategorized'],
    'data.app': 'Zen',
    'data.title': 'gorse-io/gorse: Gorse open source recommender system engine',
    'data.url': np.nan,
    'duration': 2.37,
    'id': 17857,
    'timestamp': '2025-11-01T09:28:40.700500Z'},
    {'data.$category': ['Productivity', 'Gemini'],
    'data.app': 'Zen',
    'data.title': 'Google Gemini',
    'data.url': np.nan,
    'duration': 85.522,
    'id': 17856,
    'timestamp': '2025-11-01T09:28:43.070500Z'}]

    

    return pd.DataFrame(data)


@pytest.fixture
def get_processed_data() ->pd.DataFrame:
    '''the output data of function prepare_aw_events_data which is literally the processed data of the data above '''
    
    data=[{'duration': -0.7071067811865476,
        'time_cos': -0.13629194316349444,
        'time_sin': -0.990668716690256,
        'title_category': 'category: Uncategorized | title: gorse-io/gorse: Gorse '
                            'open source recommender system engine'},
        {'duration': 0.7071067811865475,
        'time_cos': -0.13607580953541654,
        'time_sin': -0.9906984274032543,
        'title_category': 'category: Productivity,Gemini | title: Google Gemini'}
        ]

    return pd.DataFrame(data,columns=['title_category','duration','time_sin','time_cos'])




@pytest.fixture
def get_activity_watcher_encoder_input(get_processed_data) ->tuple[np.ndarray,torch.Tensor]: 
    '''get the input data of ActivityWatcherEncoder'''

    return prepare_sentence_transformer_input(get_processed_data)


@pytest.fixture
def get_activity_watcher_encoder_output() -> torch.Tensor:
    '''get the output  of activity watcher encoder(i.e. the data after encoded)'''

    output=torch.load(FIXTURE_PATH/'encode_aw_events_function_fixture.pt')

    return output

@pytest.fixture
def get_timestamp() ->pd.Series:
    '''return a batch of timestamp which is mixed with -100'''
    data=['-100', '-100', '-100', '-100', '-100',
       '2025-11-08T10:12:47.671944', '-100', '-100', '-100', '-100',
       '-100', '-100', '-100', '-100', '-100', '-100']
    
    return pd.Series(data)

@pytest.fixture
def produc_model() -> ProductiveModel:
    '''create a fresh model class'''
    config=ProductiveModelConfig

    return ProductiveModel(config).to(DEVICE)


def productive_dataset(with_y:bool=True) -> ProductiveData:
    '''give a fresh dataset class to test'''

    #this is not the correct train data model should get,but they have similar structure, so we could use it to test
    data=pd.read_csv(FIXTURE_PATH/"like_dislike_tag_data_tag's_interest=1.csv")

    #convert string number to float
    data.loc[:,'duration']=iso_duration_transform(data.loc[:,'duration']).apply(float) 

    #the data doesn't contain a productive rate, so we use interest to test it
    if with_y:
        data.loc[:,'productive_rate']= data.loc[:,'interest']
       

    else:
        data=data.drop(columns='interest')
 

    return ProductiveData(data)

@pytest.fixture
def dataset_with_y() ->ProductiveData:
    '''the productive dataset with y'''
    

    return productive_dataset()

@pytest.fixture
def dataset_without_y() ->ProductiveData:
    '''the productive dataset without y'''

    return productive_dataset(with_y=False)


@pytest.fixture
def get_encoder():
    '''get a fresh sentence transformer encoder'''

    set_random_seed()

    return ActivityWatcherEncoder().to(DEVICE)


@pytest.fixture(name='hpm_predict')
def get_fresh_hybird_productive_model_predicting_class() -> HybirdProductiveModelPredicting:
    '''just get a fresh class of HybirdProductiveModelPredicting'''

    return HybirdProductiveModelPredicting()




    

