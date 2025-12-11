import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEVICE


class ActivityWatcherEncoder(nn.Module):
    '''Activity Watcher's data encoder, a sentence transformer'''

    def __init__(self):
        super().__init__()

        self.event_encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)


    def forward(self,aw_text:np.ndarray) ->torch.Tensor:
        '''use sentence transfomer to encode aw events 
        
            Args:

                aw_text{np.ndarray}: the text part of aw data(i.e. the part('title_category') 
                    that contain the text of title and category)
                
            Returns:
                torch.Tensor

                    A torch.Tensor vector contains all the text  meaning of aw events data
                    
                    Shape: (sequence_size,1024)
    
            '''
        
        self.event_encoder.eval()


        with torch.no_grad():


            encoded_text=self.event_encoder.encode(aw_text,convert_to_tensor=True,device=DEVICE)



        return encoded_text



    







    