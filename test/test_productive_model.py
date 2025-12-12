'''the test function for productive model'''
import pytest
import torch
from transformers import AutoModel
from src.models.productive_model import ProductiveModel
from torch import nn

from src.config import DEVICE





def test__init__(produc_model):
    '''test __init__ of the class'''

    assert produc_model.bge_feature_size ==1024,'the BGE model feature size is not 1024'
    assert produc_model.bge.config._name_or_path == "BAAI/bge-m3",'the model imported is not bge-m3 or None'

    assert isinstance(produc_model.productive_layer[0],nn.Linear)," the productive output layer's first layer is not linear"
    assert isinstance(produc_model.productive_layer[1],nn.ReLU)," the productive output layer's second layer is not ReLu"
    assert isinstance(produc_model.productive_layer[2],nn.Dropout)," the productive output layer's third layer is not Dropout"
    assert isinstance(produc_model.productive_layer[3],nn.Linear)," the productive output layer's fourth layer is not linear"

    assert isinstance(produc_model.interest_layer[0],nn.Linear)," the interest output layer's first layer is not linear"
    assert isinstance(produc_model.interest_layer[1],nn.ReLU)," the interest output layer's second layer is not ReLu"
    assert isinstance(produc_model.interest_layer[2],nn.Dropout)," the interest output layer's third layer is not Dropout"
    assert isinstance(produc_model.interest_layer[3],nn.Linear)," the interest output layer's fourth layer is not linear"


def test_forward(produc_model):
    '''test test the forward part of the productive model'''
    seq_len=400
    out_feature=2
    batch_size=1

    fake_data=torch.randint(low=0,high=10,size=(batch_size,seq_len),dtype=torch.long)

    attention_mask=torch.randint(low=0,high=2,size=(batch_size,seq_len)) 



    duration=torch.randn(size=(batch_size,))

    with torch.no_grad():

        
        produc_model.eval()

        output=produc_model(input_ids=fake_data.to(DEVICE),attention_mask=attention_mask.to(DEVICE),duration=duration.to(DEVICE))


        expect_shape=(batch_size,out_feature)

        assert output['productive_rate'].shape==expect_shape,f"the output shape ({output['productive_rate'].shape}) is not matching the {expect_shape}"

        assert output['interest'].shape==expect_shape,f"the output shape ({output['interest'].shape}) is not matching the {expect_shape}"
    








