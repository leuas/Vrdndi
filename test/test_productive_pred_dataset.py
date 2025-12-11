'''the test file for productive_pred_dataset'''

import torch
import pytest
import pandas as pd


from config import FIXTURE_PATH
from Personal_feed.src.model_dataset.productive import ProductiveData
from src.utils.ops import duration_transform





def test__init__(dataset_with_y,dataset_without_y):
    '''test the __init__ part of the class'''

    assert dataset_with_y.max_length == 120,'the max length of productive dataset is not 120'
    assert dataset_with_y.tokenizer.name_or_path== "BAAI/bge-m3",'the toeknzier is not for BBAAI/bge-m3'


    #test x part
    assert isinstance(dataset_with_y.x,pd.DataFrame),'x is not dataframe'
    assert 'title' in dataset_with_y.x.columns
    assert 'description' in dataset_with_y.x.columns
    assert 'youtuber' in dataset_with_y.x.columns
    assert 'duration' in dataset_with_y.x.columns


    #test y part
    assert isinstance(dataset_with_y.y,pd.DataFrame|pd.Series),'y is not dataframe'
    assert 'productive_rate' in dataset_with_y.y.columns
    assert 'interest' in dataset_with_y.y.columns

    assert dataset_without_y.y is None,"y should be None in the dataset that doesn't have y "




def test__len__(dataset_with_y):
    '''test the __len__ function of the productive dataset'''

    assert len(dataset_with_y) == 892,"the length of the data is not 892"


def test__get_item__with_y(dataset_with_y):
    '''test the get item function with the dataset that has y'''

    single_item=dataset_with_y[0]

    assert isinstance(single_item,dict),'the item from dataset is not a dict'

    expected_keys = ['input_ids', 'attention_mask','duration', 'productive_rate','interest']

    assert all(key in single_item for key in expected_keys),\
    f"the item from dataset doesn't contain one of or some of keys from expected_keys\
    ({expected_keys}))"


    for key in ['input_ids','attention_mask']:

        tensor=single_item[key]

        tensor_dtype=tensor.dtype
        tensor_shape=tensor.shape
        tensor_shape_len=len(tensor_shape)

        assert isinstance(tensor,torch.LongTensor),f"type should be {torch.LongTensor}, but got {type(tensor)}"

        assert tensor_dtype == torch.long,f"dtype should be {torch.long}, but got {tensor_dtype} "

        assert tensor_shape_len ==1, f"expected 1 dimension, but got{tensor_shape_len}"

        assert tensor_shape[0] <=120,f"expected value is less than 120(max length), but got {tensor_shape[0]}"

   
    duration_shape=single_item['duration'].shape 
    duration_dtype=single_item['duration'].dtype

    produc_rate_shape=single_item['productive_rate'].shape
    produc_rate_dtype=single_item['productive_rate'].dtype

    interest_shape=single_item['interest'].shape
    interest_dtype=single_item['interest'].dtype

    expected_shape=torch.Size([]) #a number


    assert duration_shape == expected_shape,f"expected shape: {expected_shape} , got {duration_shape} "

    assert produc_rate_shape == expected_shape,f"expected shape: {expected_shape}, got {produc_rate_shape} "
    
    assert interest_shape == expected_shape,f"expected shape: {expected_shape}, got {interest_shape} "


    assert duration_dtype== torch.float,f"expected dtype: torch.float , got {duration_dtype} "

    assert produc_rate_dtype == torch.long,f"expected dtype: torch.long , got {produc_rate_dtype} "

    assert interest_dtype == torch.long,f"expected dtype: torch.long , got {interest_dtype} "


def test__get_item_without_y(dataset_without_y):
    '''test the get item function with the dataset that doesn't have y'''

    single_item=dataset_without_y[0]

    assert 'productive_rate' not in single_item,'productive rate(y) in the dataset that was not supposed to have y'
    assert 'interest' not in single_item,'interest(y) in the dataset that was not supposed to have y'




