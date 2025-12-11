'''contain the test function of predict_productive_model'''

import pytest
import pprint

import torch
import pandas as pd



def test_prepare_predicting_data(hpm_predict)->None:
    '''test prepare predicting data'''


    data=hpm_predict._prepare_predicting_data()
    

    assert isinstance(data,pd.DataFrame), ' The output of prepare_predicting_data should be pd.DataFrame'

    data_col=['youtuber', 'description', 'title', 'videoId', 'data_state', 'duration','upload_time', 'timestamp']

    assert data.columns == data_col,' some data columns is missed'


def test_get_preds_from_hybird_productive_model(hpm_predict) -> None:
    '''test get_preds_from_hybird_productive_model function'''

    output = hpm_predict.get_preds_from_hybird_productive_model()

    pprint.pprint(output)



    





    

    






