
import torch
import types
import torch.nn as nn
import os
from torchview import draw_graph
from src.models.productive import HybirdProductiveModel
from visualtorch import layered_view
from src.config import HybirdProductiveModelConfig,DEVICE,INFERENCE_DATA_PATH
from src.models.components import ResBlock
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class WrapperModle(nn.Module):
    '''A Wrapper for hybirdProductiveModel to make the forward input limits to one in order to use layered_view'''

    def __init__(self,model) -> None:
        super().__init__()

        self.dummy_aw_tensor=torch.randn(4,2000,768).cpu()

        self.dummy_aw_mask=torch.randn(4,2000).cpu()

        self.dummy_duration=torch.randn(4,1).cpu()
        self.attention_mask=torch.randint(high=100,size=(4,300)).long().cpu()

        self.x=torch.randint(high=100,size=(4,300)).long().cpu()

        self.model=model


    def forward(self,x):
        '''forward process of the Wrapper model , feed the x to the hybirdproductivemodel'''

        

        output=self.model(self.x,self.attention_mask,self.dummy_duration,self.dummy_aw_tensor,self.dummy_aw_mask)


      

        return output




def hg_size(self:BaseModelOutputWithPastAndCrossAttentions):
    '''return the transformer output size in huggingface way to aovoid the mismatch
      between visualtorch's expected tensor and hg's BaseModelOutputWithPastAndCrossAttentions '''
    
    return self.last_hidden_state.size()



def patched_forward(self,*args,**kwargs):
    '''patch the forward part of real model to output list of tensor instead of dict'''

    rs=ori_forward(*args,**kwargs)

    return [rs['interest'].unsqueeze(-1),rs['productive_rate'].unsqueeze(-1)]




resblock=ResBlock(cond_dim=1)



dummy_aw_tensor=torch.randn(4,2000,768).cpu()

dummy_aw_mask=torch.randn(4,2000).cpu()
dummy_duration=torch.randn(4,1).cpu()
attention_mask=torch.randint(high=100,size=(4,300)).long().cpu()

x=torch.randint(high=100,size=(4,300)).long().cpu()

input=(x,attention_mask,dummy_duration,dummy_aw_tensor,dummy_aw_mask)

model=HybirdProductiveModel(HybirdProductiveModelConfig())

#graph=draw_graph(model=model,input_data=input)


#graph.visual_graph.render("productive_main_model_structure", format="png")
seq_compressor_input=(dummy_aw_tensor,dummy_duration)

anograph=draw_graph(model=model.sequence_compressor,input_data=seq_compressor_input)

anograph.visual_graph.render("Custom_residual_block_structure",format='png')