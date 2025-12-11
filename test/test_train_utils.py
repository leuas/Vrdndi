import pytest
import numpy as np
import pandas as pd
import torch
import pprint

from src.utils.ops import calc_bce_posweigt,combine_aw_title_category,prepare_aw_events_data,pad_aw_sequence

from src.config import FIXTURE_PATH



@pytest.mark.parametrize("label2,posweight2",[
    ([0, 1, 0, 1, 0, 1],1.0),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1],9.0),
    ([1, 1, 1, 0], 1 / 3), 
    ([0, 0, 0, 0],4.0), #handle zero division 
])

def test_calc_bce_posweight_another_way(label2,posweight2) -> None:
    '''test the calc bec psoweight function'''

    assert calc_bce_posweigt(label2) ==pytest.approx(posweight2)


def test_combine_title_category() ->None:
    '''test the combine title category function'''
    
    #fixture
    data=pd.read_csv(FIXTURE_PATH/'aw_app_sequence_test.csv')

    output=combine_aw_title_category(data)

    #TODO:there's bug, see notion,report bug:test_combine_title_category str joint error

    row_num_in=data.shape[0]
    row_num_out=output.shape[0]


    assert isinstance(output,pd.Series),f'expected pd.Series, got {type(output)} instead'

    assert row_num_in == row_num_out, f"numbers of rows in input and output doesn't match, \
        got num of row in input: {row_num_in}; in output: {row_num_out}"
    
    assert isinstance(output.loc[0],str),f'the combined title and category is not str in the output,\
        got type: {type(output.loc[0])} insetad'
    
    

def test_prepare_aw_events_data(mocker,get_aw_data,get_processed_data,get_time) ->None:
    '''test the prepare aw events data'''

    mock_get=mocker.patch('src.utils.ops.get_aw_raw_data',return_value=get_aw_data)


    output=prepare_aw_events_data(end_time=get_time)


    mock_get.assert_called_once_with(
        end_time=get_time,
        hours=24,
        hostname='leuasMacBook-Air.local'
    )


    pd.testing.assert_frame_equal(output,get_processed_data)




def test_encode_aw_events(get_activity_watcher_encoder_input,get_encoder) ->None:
    ''''Test the encode aw events function'''


    model=get_encoder

    aw_text,_=get_activity_watcher_encoder_input

    output=model(aw_text)


    expected_output=torch.load(FIXTURE_PATH/'encode_aw_events_function_fixture.pt')


    torch.testing.assert_close(output.detach().cpu(),expected_output.cpu())




def test_pad_aw_sequence(get_activity_watcher_encoder_output) ->None:
    '''test the pad_aw_sequence_data'''
    #Here I only use two timestamp as fixture to test, 
    # which may less robust than using real timestamp series which contains some -100


    aw_tensor,aw_attention_mask=pad_aw_sequence(get_activity_watcher_encoder_output,384)


    assert isinstance(aw_tensor,torch.Tensor),f'aw_tensor expected type: torch.Tensor, but got {type(aw_tensor)} instead'
    assert isinstance(aw_attention_mask,torch.Tensor),f'aw_attention mask expected type: torch.Tensor, but get{type(aw_attention_mask)} instead'

    aw_tensor_shape=aw_tensor.shape
    aw_mask_shape=aw_attention_mask.shape

    aw_tensor_dim_num=len(aw_tensor_shape)

    assert aw_tensor_dim_num==2,f'Expected aw_tensor has two dimension, got {aw_tensor_dim_num} insetad'

    assert aw_tensor_shape==(2,1024),f'Expected aw_tensor has shape of (2,1024), got {aw_tensor_shape} instead )'

    assert aw_tensor_shape== aw_mask_shape,'Expected aw_tensor has same shape with aw_attention_mask'

   






    



    

