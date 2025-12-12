''' The file contain the text function of dat_etl.py'''

import pytest
import pandas as pd
import pprint
import numpy as np
from datetime import datetime


from src.utils.data_etl import convert_timezone,get_aw_raw_data


def test_convert_time_zone()->None:
    '''test the convert time zone function'''

    time_str='2025-11-08T09:31:05.401000+00:00'


    output=convert_timezone(time_str)

    assert isinstance(output,(pd.Series,datetime)),f'Expected pd.Series, got {type(output)} instead '


def test_get_aw_raw_data(get_time)->None:
    '''test the get_aw_raw_data function'''

    
    output=get_aw_raw_data(end_time=get_time)

    assert isinstance(output,pd.DataFrame), f'Expected pd.Dataframe, got {type(output)} instead '

    column=['duration','id','timestamp','data.app','data.$category','data.title','data.url']

    assert set(column).issubset(output.columns),f'Expected columns: {column}, got {output.columns} instead'














    


