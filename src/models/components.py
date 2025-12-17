'''Custom layer or block that used in Productive model'''
import logging
import torch

import torch.nn as nn


class Permute(nn.Module):
    '''A helper class for using permute between different block or layer in Sequential'''

    def __init__(self,):
        super().__init__()

    def forward(self,tensor:torch.Tensor) ->torch.Tensor:
        '''use permute to change the shape of tensor'''

        return tensor.permute(0,2,1)



@torch.jit.script
def fused_compute_resblock_output(output:torch.Tensor,residual:torch.Tensor):
    ''''compute the final output of residual block'''

    return output+residual


class ResBlock(nn.Module):
    '''A Residual block for processing equential data (i.e. aw event sequence) 
        
        The architecture follows the structure:
        Input -> [[ AdaLN -> SiLU -> Conv1d -> Dropout -> AdaLN -> SiLU -> Conv1d ] -> SE] + Input 
        
        Args:
            in_ch{int}: The number of input_channels
            out_ch{int}: The number of output_channels
            kernel_size{int}: The size of kernel in convolution layer. Default to 7
            stride{int}: The size of stride in convolution layer. Default to 3
            dropout_pro{float}: Probability of an input element to be zeroed. Default to 0.3
            cond_dim: The number of condtion dimension
            '''

    def __init__(self, in_ch:int=1024,out_ch:int=1024,kernel_size:int=7,stride:int=3,
                dropout_prob:float=0.3,cond_dim:int|None=None) -> None:
        #after the projector, token size is 1024, so in_ch is 1024 in default
        super().__init__()

        padding=(kernel_size-1)//2

        self.seblock=SEBlock(out_ch)

        self.resblock=ConditionAwareSequential(
            
            #Downsampling
            ModulateAdaLN(in_ch,cond_dim),
            nn.SiLU(),
            nn.Conv1d(in_ch,out_ch,kernel_size,stride,padding=padding,bias=False),
            nn.Dropout(dropout_prob),
           
            #Refining
            ModulateAdaLN(out_ch,cond_dim),
            nn.SiLU(),
            nn.Conv1d(out_ch,out_ch,kernel_size,stride=1,padding=padding,bias=False),

        
        )


        self.downsample=None

        #downsample the original tensor 
        if stride >1 or in_ch !=out_ch:
            self.downsample=nn.Sequential(
    
                nn.Conv1d(in_ch,out_ch,kernel_size=1,stride=stride,bias=False),

                Permute(),
                nn.LayerNorm(out_ch),# Shape: (Batch, Sequence_size, Token_size)
                Permute()

            )

    def forward(self,inputs:torch.Tensor,condition:torch.Tensor) ->torch.Tensor:
        '''Process sequence through the residual block 
            
            Args:
                inputs{torch.Tensor}: Input tensor of shape (Batch, In_channel, Sequence_size)
                condition (torch.Tensor): Global conditioning vector. 
                    Shape: [Batch_Size, Cond_Dim]

                
            Returns:
                torch.Tensor: Output tensor of shape(Batch, Out_channels, Length) '''
        

        residual=inputs

        output=self.resblock(inputs,condition)

        if self.downsample is not None:
            residual= self.downsample(inputs)

        weighted_output=self.seblock(output)

        final_output=fused_compute_resblock_output(weighted_output,residual)

        return final_output


class SEBlock(nn.Module):
    '''Squeeze-and-Excitation (SE) Block for the sequence events data of AcivityWatcher (AW) in the ResBlock
       

        Args:
            channels{int}: The number of input channels

            reduction:{int, optional}: The reduction ratio for the bottleneck. 
                A higher value would save more parameters, but compress information more.  Default to 16

        '''
    def __init__(self,channels:int,reduction:int=16):
        super().__init__()

        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.seblock=nn.Sequential(
            #compressing
            nn.Linear(channels,channels//reduction,bias=False),
            
            nn.ReLU(inplace=True),#override the original vector
            #expanding
            nn.Linear(channels//reduction,channels,bias=False),
            nn.Sigmoid()#get the important score
        )

    def forward(self,ori_tensor:torch.Tensor) ->torch.Tensor:
        '''Combine the important score with original tensor'''

        b,c,_=ori_tensor.size() #(batch_size, channels, sequence_size)

        compressed_tensor=self.avg_pool(ori_tensor).view(b,c)

        important_score=self.seblock(compressed_tensor).view(b,c,1)

        return ori_tensor*important_score
    

@torch.jit.script
def adaln_modulation(input_tensor:torch.Tensor,gamma:torch.Tensor,beta:torch.Tensor) ->torch.Tensor:
    '''calculate modulating normalized input part from AdaLN  '''

    return input_tensor*(1+gamma)+beta



class AdaLN(nn.Module):
    '''Adaptive Layer Normalization with Zero-Initialization for adjusing the token from 
        sequence_compressor in HybridProductiveModel
    
        This layer modulates the normalized input using scale (gamma) and shift (beta)
        values predicted from a condition vector (e.g., duration or time embedding).

        Args:
            ch (int): The dimension of the input sequence features.

            cond_dim (int, optional): The dimension of the conditioning vector (e.g., duration). 
                If cond_dim is None (default), this layer would act like normal Layer Normalization
    
    '''

    def __init__(self, ch:int, cond_dim:int|None=None):
        super().__init__()

        self.norm=nn.LayerNorm(ch,elementwise_affine=False,eps=1e-6)



        if cond_dim is None:
            self.if_cond=False
            logging.warning("WARNING: cond_dim is None. The condition adapter is disabled;" \
            " AdaLN will function as standard LayerNorm")
        
        else:
            self.if_cond=True
            
            self.condition_adapter=nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim,2*ch,bias=False)
            )


            nn.init.constant_(self.condition_adapter[-1].weight,0)

    def forward(self,input_tensor:torch.Tensor,condition:torch.Tensor):
        '''
        Applies normalization and modulation.

        Args:
            input_tensor (torch.Tensor): Input sequence. 
                Shape: [Batch_Size, Token_Len, Sequense_size]
            condition (torch.Tensor): Global conditioning vector. 
                Shape: [Batch_Size, Cond_Dim]

        Returns:
            torch.Tensor: Modulated sequence with same shape as input 'x'.
                Shape: [Batch_Size, Token_Len, Sequense_size]
        '''

        input_tensor=self.norm(input_tensor)


        if self.if_cond:

            if condition.dim() ==1: #(Batch, ) -> (Batch, 1)
                condition=condition.unsqueeze(-1)

            style=self.condition_adapter(condition)

            style=style.unsqueeze(1) #(Batch, 1 , 2 * Token_size of input_tensor)

            gamma,beta=style.chunk(2,dim=-1)

            return adaln_modulation(input_tensor,gamma,beta)
        
        return input_tensor


class ModulateAdaLN(nn.Module):
    '''
    Wraps the Permute -> AdaLN -> Permute logic
    '''

    def __init__(self, ch:int, cond_dim:int|None):
        super().__init__()

        self.norm=AdaLN(ch,cond_dim)

    def forward(self,input_tensor:torch.Tensor,condition:torch.Tensor) ->torch.Tensor:
        '''apply the wraping logic'''

        input_tensor= input_tensor.permute(0,2,1)

        output=self.norm(input_tensor,condition)

        output=output.permute(0,2,1)

        return output


class ConditionAwareSequential(nn.Sequential):
    '''
    A generic Sequential container that supports conditional inputs (e.g., AdaLN).

    Unlike standard nn.Sequential which blindly passes one argument, this class
    checks the type of each layer during the forward pass. If the layer is a
    conditional normalization layer (AdaLN or ModulateAdaLN or Residual Block), 
    it passes both the input tensor and the condition vector.

    Args:
        *args: Comma-separated list of child modules (same as nn.Sequential).
    '''

    def forward(self,input_tensor:torch.Tensor,condition:torch.Tensor) ->torch.Tensor:
        """
        Forward pass with dynamic argument dispatch.

        Args:
            input_tensor (torch.Tensor): Input sequence.
            condition (torch.Tensor): The global condition vector (e.g., duration).

        Returns:
            torch.Tensor: The processed output.
        """
        
        for module in self:
            if isinstance(module,(AdaLN, ModulateAdaLN, ResBlock)): #If it's AdaLN or related class, input two args
                input_tensor = module(input_tensor,condition)

            else: 
                input_tensor = module(input_tensor)

        return input_tensor



class SwiGLU(nn.Module):
    """
    A Gated Linear Unit (SwiGLU) designed for dimensionality reduction.
    
    This layer replaces a standard 'Linear -> Activation' block. It projects
    the input to a larger dimension, splits it into a gate and a value,
    and multiplies them to produce a compressed output.

    Args:
        in_features (int): Size of each input sample. 
        out_features (int): Size of each output sample.
        hidden_feature (int,optional): Size of intermediate vector. Default to 2 * out_feature
    """

    def __init__(self,in_feature:int,out_feature:int,hidden_feature:int|None=None ):
        super().__init__()

        if hidden_feature is None:
            hidden_feature=out_feature*2 

        self.gate_and_value_proj=nn.Linear(in_feature,hidden_feature *2,bias=False) # "*2" for two vector

        self.output_proj=nn.Linear(hidden_feature,out_feature,bias=False)

    def forward(self,input_tensor:torch.Tensor) ->torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Input tensor. 
                Shape: (Batch_Size, in_features)

        Returns:
            torch.Tensor: Compressed and activated output.
                Shape: (Batch_Size, out_features)
        """

        combined=self.gate_and_value_proj(input_tensor)

        gate,value=combined.chunk(2,dim=-1) 

        hidden=nn.functional.silu(gate) *value

        return self.output_proj(hidden)
