'''Training part of the Recursive model'''

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

from src.config import RecursiveBGEConfig,DEVICE
from src.models.recursive_bge_m3 import DistillRecursiveModel
from datasets import load_dataset

class RecursiveBGETraining:
    '''training part of recursive model'''

    def __init__(self,config:RecursiveBGEConfig) -> None:
        

        self.config=config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ori_model_name)

        self.ori_model=AutoModel.from_pretrained(self.config.ori_model_name).to(DEVICE)

        self.distill_model=DistillRecursiveModel(
            model_name=self.config.ori_model_name,
            max_steps=self.config.max_steps,
            init_layer_index=self.config.init_layer_index
            ).to(DEVICE)
        
        self.optimizer = AdamW(self.distill_model.parameters(), lr=self.config.lr)
        self.loss_fn = nn.CosineEmbeddingLoss()
        

    def get_training_data_iterator(self):
        # 1. Load mC4 (Multilingual) - Streaming mode so it doesn't fill your disk
        # 'en' is English, you can mix in 'zh', 'fr', etc.
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        
        # 2. Infinite Loop Generator
        # This creates a never-ending stream of sentences for your training loop
        for row in dataset:
            text = row['text']
            
            # Filter out garbage (too short/too long)
            # In case the sentence is longer than the limitation of sequence size
            if len(text) > 50 and len(text) < 2000:
                yield text



    def train(self,save_name:str="bge_recursive.pth"):
        '''start training '''

        print("Starting Training Loop...")
        data_stream = self.get_training_data_iterator()

        # --- THE LOOP ---
        for epoch in range(self.config.total_epoch):
            total_loss = 0
            batch_texts = []
        
            # Manually fill the batch from the stream
            for _ in range(self.config.batch_size):
                try:
                    # Grab next sentence from internet/dataset
                    sentence = next(data_stream) 
                    batch_texts.append(sentence)
                except StopIteration:
                    break
            
            # Simple batching (In production, use torch.utils.data.DataLoader)
            for i in range(0, len(batch_texts), self.config.batch_size):
                batch_texts = batch_texts[i : i + self.config.batch_size]
                
                # A. Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=8192, 
                    return_tensors="pt"
                ).to(DEVICE)

                # B. Get Teacher's "Gold Standard" Output
                with torch.no_grad(): # Don't calculate gradients for Teacher! Saves RAM.
                    teacher_out = self.ori_model(**inputs)
                    # BGE-M3 uses the first token [CLS] (index 0)
                    teacher_vec = teacher_out.last_hidden_state[:, 0]
                    # Normalize Teacher (Crucial!)
                    teacher_vec = torch.nn.functional.normalize(teacher_vec, p=2, dim=1)

                # C. Get Student's Output
                # We pass input_ids and attention_mask.
                student_vec, ponder_cost = self.distill_model(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'],
                    tau=self.config.tau
                )
                # Note: Student vector is ALREADY normalized inside the model class we wrote.

                # D. Calculate Loss
                # 1. Semantic Loss: "Point in the same direction"
                # We create a target of [1, 1, 1...] dynamically using the teacher's device
                target_ones = teacher_vec.new_ones(teacher_vec.size(0))
                similarity_loss = self.loss_fn(student_vec, teacher_vec, target_ones)

                # 2. Efficiency Loss: "Don't think too long"
                # We combine them. 
                loss = similarity_loss + ponder_cost

                # E. Backpropagation
                self.optimizer.zero_grad() # Clear old gradients
                loss.backward()       # Calculate new gradients
                self.optimizer.step()      # Update Student weights

                # Logging
                total_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f} | Sim: {similarity_loss.item():.4f} | Ponder: {ponder_cost.item():.4f}")

        # 7. Save the Model
        print("Saving Student...")
        torch.save(self.distill_model.state_dict(), save_name)
        print("Done!")




