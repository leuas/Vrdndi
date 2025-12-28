'''the recursive BGE-M3 model'''

import logging
import torch

import torch.nn as nn
from src.models.components import RecursiveACTLayer
from transformers import AutoModel, AutoConfig


class DistillRecursiveModel(nn.Module):
    '''recursive small BGE-M3'''

    def __init__(self,model_name:str="BAAI/bge-m3",max_steps:int=36,init_layer_index:int=0) -> None:
        super().__init__()

        print(f"Loading weights from {model_name}...")
        # 1. Load the Config (to get hidden size, vocab size, etc.)
        self.config = AutoConfig.from_pretrained(model_name)
        hidden_dim = self.config.hidden_size
        
        # 2. Load the ORIGINAL Model temporarily to steal its parts
        # (We use CPU to save GPU memory for the training)
        original_model = AutoModel.from_pretrained(model_name).cpu()
        
        # 3. STEAL THE EMBEDDINGS (The "Mouth")
        # This copies the massive multi-lingual dictionary
        self.embeddings = original_model.embeddings

        pretrained_block = original_model.encoder.layer[init_layer_index]
        
        # 4. CREATE THE NEW BRAIN (Recursive ACT)
        # We use a standard Transformer Encoder Layer inside our loop
        self.recursive_brain = RecursiveACTLayer(hidden_dim, max_steps=max_steps,pretrained_layer=pretrained_block)
        
        # 5. Clean up
        del original_model # Free up RAM
        print("Initialized Tiny Recursive small BGE-M3 model!")


    def forward(self, input_ids, attention_mask, tau=0.01):
            # 1. Get Initial Embeddings (The "Mouth")
            # Input: (Batch, Seq_Len) -> Output: (Batch, Seq_Len, 1024)
            x = self.embeddings(input_ids)

            # 2. Run the Recursive Brain (The "Loop")
            # We pass the raw attention_mask (0s and 1s). 
            # Your RecursiveACTLayer will handle the -10000 conversion internally.
            x, ponder_cost = self.recursive_brain(x, attention_mask=attention_mask, tau=tau)

            # 3. Pooling (Extract the Sentence Representation)
            # BGE-M3 (RoBERTa) uses the [CLS] token at index 0 to represent the whole sentence.
            sentence_embedding = x[:, 0] 

            # 4. Normalization (Crucial for Cosine Loss!)
            # This makes sure the vector length is 1.0, so the loss only measures direction.
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

            return sentence_embedding, ponder_cost
