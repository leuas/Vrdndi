'''this file contain training function of baseline model'''

import pickle
import pprint
import numpy as np
import pandas as pd



from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report


from src.utils.data_etl import prepare_rf_data



#------------------------------PART THREE:BASELINE MODEL--------------------------------------




#TODO if you wanna know which video you watched or not, you need to make a database yourself

def log_regression_train():
    '''split data into train and test set(0.75:0.25) and train the logistic regression model'''

    x=pd.read_csv('x.csv')
    y=pd.read_csv('y.csv')
    
    title_col='title'
    description_col='description'
    youtuber_col='youtuber'

    #check and replace the empyt palce with '' in input column
    x[title_col]=x[title_col].fillna('')
    x[description_col]=x[description_col].fillna('')
    x[youtuber_col]=x[youtuber_col].fillna('')


    print(x.shape,y.shape)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=36)

    #flatten the y
    y_test,y_train=y_test.values.ravel(),y_train.values.ravel()

    #vectorize each column
    preprocessor=ColumnTransformer(
        transformers=[
            ('title_vec',TfidfVectorizer(),title_col),
            ('descrip_vec',TfidfVectorizer(),description_col),
            ('youtuber_vec',TfidfVectorizer(),youtuber_col)
        ],
        remainder='drop')
    
    classifier=LogisticRegression()

    model_pipline=Pipeline([('preprocessor',preprocessor),('classifier',classifier)])
    
    model_pipline.fit(x_train,y_train)
    
    accuracy=model_pipline.score(x_test,y_test)

    print(accuracy)
    with open('logsitic_regression.pkl','wb') as file:
        pickle.dump(model_pipline,file)
    


def random_forest_train():
    '''train the random forest'''

    train_x,test_x,train_y,test_y=prepare_rf_data()

    preprocessor=ColumnTransformer(
        transformers=[
        ('title_vec',TfidfVectorizer(),'title'),
        ('description_vec',TfidfVectorizer(),'description'),
        ('youtuber_vec',TfidfVectorizer(),'youtuber')],
        remainder='passthrough',
    )

    classifer=RandomForestClassifier(random_state=10,class_weight='balanced')

    model_pipline=Pipeline([('preprocessor',preprocessor),('classifier',classifer)])

    model_pipline.fit(train_x,train_y)

    prediction=model_pipline.predict(test_x)

    print(accuracy_score(test_y,prediction))

    print(classification_report(test_y,prediction))

    with open('randomforest.pkl','wb') as file:
        pickle.dump(model_pipline,file)


if __name__ =='__main__':
    random_forest_train()