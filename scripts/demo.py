'''Model demo'''
import pprint
import pandas as pd

from src.db.database import VrdndiDatabase
from src.inference.productive import HybirdProductiveModelPredicting

class Demo:
    '''A demo for model performance showcase'''

    def __init__(self) -> None:
        
        self.db=VrdndiDatabase()

        self.inference=HybirdProductiveModelPredicting()
    def get_unique_items(self) ->None:
        '''Find the datapoint that belong to same itemds but has different feedback (in different time)
        '''


        feedback=self.db.get_feedback()

        unique_counts=feedback.groupby('videoId')['productive_rate'].transform('nunique')

        mask=unique_counts>1

        pprint.pprint(feedback[mask])

    def predict_feedback(self):
        '''use model to predict feedback in different time'''

        data=self.db.get_feedback()

        self.inference.predict()

        