'''the script run the training logic of productive model'''

import os
import logging
from src.utils.ops import interest_productive_data_preprocess,timestamp_data_preprocess
from src.pipelines.productive import HybridProductiveModelTraining
from src.config import HybridProductiveModelConfig
from src.path import TRAIN_DATA_PATH


if __name__=='__main__':

    logging.basicConfig(
        level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
    )
    config=HybridProductiveModelConfig()
    
    test1=HybridProductiveModelTraining(config=config)


    test1.start_train(model_name='test.pth')
    
    
    #test1.kfold_start()
    