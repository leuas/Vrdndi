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
    '''Squeeze-and-Excitation (SE) Block for the sequence events data of AcivityWatch (AW) in the ResBlock
       

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



class RecursiveACTLayer(nn.Module):
    '''
    A recurdive ACT layer by using a pretrained_layer or the transformer layer it create 
    to recursively process the data
    
    Args:
        hidden_dim(int): THe hidden dimension of input,ouput channel of the pretrained
            layer or transformer layer. Default to 1024, which is the hidden dimension 
            of BGE-M3.
        
        max_steps(int): The maximal number of layer that model is allowed to go through.
            Default to 36, which is 3/2 times larger than regular BGE-M3 encoder layers (24).

        pretrained_lyaer: The pretrained hugging face layer. Default to None
    
    
    '''

    def __init__(self,hidden_dim:int=1024,max_steps:int=36,pretrained_layer=None) -> None:
        super().__init__()

        self.max_steps=max_steps

        self.hidden_dim=hidden_dim

        if pretrained_layer is not None:
            self.process_block=pretrained_layer
            self.is_huggingface_layer=True

        else:
            self.process_block=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=16,
                dim_feedforward=hidden_dim*4,
                activation="gelu",
                batch_first=True,
            )
            self.is_huggingface_layer=False

        
        self.model_decide_if_continue=nn.Linear(hidden_dim,1)

        # encourages model to think a bit longer(one more times)
        self.model_decide_if_continue.bias.data.fill_(1.0)

    def _update_confident_prob(self,state:torch.LongTensor,accumulation_prob:torch.Tensor) ->torch.Tensor:
        ''' Get the confident prob from model, and compare the remaining available prob
        If current confident prob plus the prob already has is larger than one, then 
        use the remaining prob instead of the confident prob from the model.
        
        Args:
            state(torch.LongTensor): The input embedding tensor or the processed tensor 
                in each step
            accumulation_prob (torch.Tensor): The accumulated confident probability accross
                the past step the model took


        Returns:
            The confident probability for this step
        '''

        decision_state=state[:,0.:]

        #How confident the model thinks the current result is.
        actual_confident_prob=torch.sigmoid(self.model_decide_if_continue(decision_state)).unsqueeze(-1)

        #The remaining probability for updating the result
        remain_confident_prob=1.0-accumulation_prob
        #If the model is confident enough for the result
        is_confident= (accumulation_prob+actual_confident_prob) >1.0

        confident_prob=torch.where(is_confident,remain_confident_prob,actual_confident_prob)

        return confident_prob


    def forward(self,input_tensor:torch.LongTensor,attention_mask:torch.LongTensor,tau:float=0.01) -> tuple[torch.Tensor,torch.Tensor]:
        '''
        Args:
            input_tensor(torch.LongTensor): The input embeddings (Batch,Seq_Len,HIdden_DIm)
            attention_mask (torch.LongTensor): Mask, 1 for words, 0 for padding. (Batch,Seq_Len)

            tau(int): The step cost (ponder cost) penalty
        '''
        batch_size=input_tensor.size(0)

        accumulation_prob=input_tensor.new_zeros(batch_size,1)
        total_step_cost=input_tensor.new_zeros(batch_size,1)
        weighted_output=torch.zeros_like(input_tensor)
        #1 = continue to process with another layer, 0 = stop
        active_layer_mask=input_tensor.new_ones(batch_size,1) 

        state=input_tensor
        for _ in range(self.max_steps):
            
            confident_prob=self._update_confident_prob(state,accumulation_prob)

            confident_prob_expanded=confident_prob.unsqueeze(-1)
            active_layer_mask_expanded=active_layer_mask.unsqueeze(-1)
            
            #update the output by plusing weighted current state IF current layer are permitted to process by model
            weighted_output+=confident_prob_expanded*state*active_layer_mask_expanded

            accumulation_prob+= confident_prob *active_layer_mask
            total_step_cost+=active_layer_mask

            confident_enough=(accumulation_prob >= (1.0-1e-6))
            #If the model is confident enough for current result, set the active_layer_mask to 0
            active_layer_mask= torch.where(confident_enough,torch.zeros_like(active_layer_mask),active_layer_mask)

            #If all sequences data in current batch are finished, break the for loop
            if active_layer_mask.sum() == 0:
                break

            if self.is_huggingface_layer:
                extended_mask=attention_mask[:,None,None,:]
                extended_mask=(1.0 -extended_mask) * -10000.0
                
                #Hugging Face returns a tuple (hidden_state,attention_map)
                layer_output=self.process_block(state,extended_mask)[0]

            else:
                layer_output=self.process_block(state,src_key_padding_mask=attention_mask)

            new_state=layer_output

            active_layer_mask_seq=active_layer_mask.unsqueeze(-1)

            state=(active_layer_mask_seq*new_state) + ((1-active_layer_mask_seq) *state )


        penalty = total_step_cost.mean() * tau

        
        return weighted_output, penalty



            
                
                    

            






            
            




