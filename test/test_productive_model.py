'''the test function for productive model'''
import pytest
import torch
from torch import nn

from src.config import DEVICE



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
    








