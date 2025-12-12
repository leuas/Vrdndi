'''this file contain the helper function for model training '''
import os
import random
import pandas as pd
import numpy as np
import torch
import copy
import wandb
import pprint
import json

from typing import Optional

from pathlib import Path

from datetime import datetime,timezone

from matplotlib import pyplot as plt

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split

from src.db.database import PersonalFeedDatabase

from src.models.activity_watcher_encoder import ActivityWatcherEncoder
from src.utils.data_etl import get_video,clean_yt_video,get_aw_raw_data,convert_timezone,get_video_data,videoid_split,duration_transform,iso_duration_transform
from src.config import DEVICE,HOSTNAME,HOST,PORT
from src.path import ARTIFACTS_PATH

import wandb


class FocalLoss(Module):
    '''
    Computes Focal Loss to fix class imbalance by focusing on hard examples.
    Args:
        alpha (float): Weighting factor to balance positive/negative classes. Default to None.
        gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0.
        ignore_index (int): Target value that is ignored 
        

    
    '''

    def __init__(self, gamma:float=2.0, alpha:torch.Tensor|None=None,ignore_index:int=-100) -> None:
        super().__init__()

        self.gamma=gamma
        self.alpha=alpha
        self.ignore_index=ignore_index


    def forward(self,input,target):
        '''the process that calculate the cross encropy loss and apply Focal in it'''

        ce_loss=torch.nn.functional.cross_entropy(
            input=input,
            target=target,
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index)
        
        pt=torch.exp(-ce_loss)

        focal_term=(1-pt)**self.gamma

        loss=focal_term * ce_loss

        final_loss=loss.mean()

        return final_loss
    





#----------------------------------------------------------------------------------------
#------------------------Helper function for model itself -------------------------------
#----------------------------------------------------------------------------------------




#NOTE currently no file use this function except some script in the legacy, remove it later
def calc_bce_posweigt(target_column:pd.DataFrame|pd.Series)->float :
    '''calculate posweight of CEL for binary classification
    Args:
        target_column: the column of data(or y; type: dataframe or series) that you want to calculate its posweight'''

    try:
        if not isinstance(target_column,(pd.DataFrame,pd.Series,pd.Index)):
            data=pd.DataFrame(target_column,columns=['random_col'])
            target_column=data['random_col']

    except (ValueError,TypeError) as e:
        raise TypeError(f'expected {pd.DataFrame,pd.Series,dict,list}, but got {type(target_column)}') from e
    
    mask = target_column != -100

    masked_col=target_column[mask]

    pos_counts=(masked_col == 1).sum()
    neg_counts=(masked_col == 0).sum()
    

    pos_weight=neg_counts / max(1,pos_counts)

    print('pos_weight',type(pos_weight))

    return pos_weight




def set_random_seed(seed:int = 666) -> None:
    '''set the random seed for random,numpy,torch mps,os,sklearn'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


    elif torch.cuda.is_available():
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False



def print_parameter_state(model:torch.nn.Module) ->None:
    '''check if the parameter is frozen or not'''

    total_params=sum(p.numel() for p in model.parameters())
    trainable_parames=sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'total params: {total_params}')
    print(f'trainable_params: {trainable_parames}')

    if total_params == trainable_parames:
        print('full model training! ')

    else:
        print("there's frozen layer")

def if_load_model(model:torch.nn.Module,model_name:str,path=ARTIFACTS_PATH,lora:bool=False) ->torch.nn.Module :
    '''load the model '''

    absolute_model_path=path/model_name
    

    state_dict=torch.load(f'{absolute_model_path}',map_location=DEVICE)
   

    if not lora:
        model.load_state_dict(state_dict)
        print(f'{model_name} model loaded successfully')

    else:
        missing,unexpected_key=model.load_state_dict(state_dict,strict=False)

        if len(unexpected_key)>0:
            print(f'WARNING: The following keys were in the file, but ignored by the model:{unexpected_key}')

        else:
            print('All the keys which are in the file are in the model')

        lora_missing=[k for k in missing if 'lora ' in k]

        if len(lora_missing)>0:
            print(f"WARNING: Model expected these LORA weights, but the file didn't have them: {lora_missing}")
        else:
            print('All the LORA layers were updated')

    return model




def text_col_fillna(x: pd.DataFrame) -> pd.DataFrame:
    '''fill nan with '' in the x'''
    assert isinstance(x,pd.DataFrame),"input x's type is not dataframe"

    text_col = ['youtuber', 'title', 'description']

    x[text_col] = x[text_col].fillna('') #since na_filter is on

    return x


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame|None]:
    '''split the data into x,y and fillna in x's text column'''
    assert isinstance(data,pd.DataFrame), "input data's type is not dataframe"

    y_label = ['productive_rate','interest']

    if set(y_label).issubset(data.columns):
        
        x = data.drop(columns = y_label)
        y = data[y_label]
        
    else:
        x=data
        y= None
    
    x=text_col_fillna(x)

    return x, y



def data_split(data:pd.DataFrame,seed:int=42
               ) ->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''split the data into train,test,validation set
    Returns:
        pd.DataFrame, in order of train_set,val_set,test_set
    '''
    idx = data.index.to_numpy().reshape(-1,1)
    train_valtest_ratio = 0.2
    val_test_ratio = 0.1


    train_idx, eval_test_idx = train_test_split(idx, test_size = train_valtest_ratio,random_state=seed)

    val_idx,test_idx = train_test_split(eval_test_idx,test_size= val_test_ratio,random_state=seed)

    train_set=data.loc[train_idx.flatten()]
    val_set=data.loc[val_idx.flatten()]
    test_set=data.loc[test_idx.flatten()]
    


    return train_set,val_set,test_set



def iterative_data_split( x:pd.DataFrame, y:pd.DataFrame
                         ) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''split the data into train, validation, test set iteratively'''

    idx = x.index.to_numpy().reshape(-1,1)

    #train: validation: test = 6: 2: 2
    train_valtest_ratio = 0.4
    val_test_ratio = 0.5

    #split the index
    train_idx, y_train_tem, x_eval_test_idx, y_val_test_tem = iterative_train_test_split(idx, y.values, test_size = train_valtest_ratio)
    test_idx, y_test_tem, val_idx, y_val_tem = iterative_train_test_split(x_eval_test_idx, y_val_test_tem, val_test_ratio)

    x_train=x.loc[train_idx.flatten()]
    x_test=x.loc[test_idx.flatten()]
    x_val=x.loc[val_idx.flatten()]
    
    y_train=y.loc[train_idx.flatten()]
    y_test=y.loc[test_idx.flatten()]
    y_val=y.loc[val_idx.flatten()]


    return x_train,x_val,x_test,y_train,y_val,y_test


#----------------------------------------------------------------------------------------
#------------------------Part of processing Activity Wachter's data----------------------
#----------------------------------------------------------------------------------------




def combine_aw_title_category(data:pd.DataFrame) ->pd.Series:
    '''combine the aw title and category
        
        Args:
            data: the aw data after json normalized '''

    #concate the category list into a string,i.e. 'Productivity,Programming,Projects'

    category=data['data.$category'].str.join(',')



    title=data['data.title']


    data['combined_str']='category: '+category+' | title: '+title

    return data['combined_str']


def converted_aw_timestamp(timestamp:pd.DataFrame|pd.Series) -> pd.DataFrame:
    '''convert the timstamp to cos and sin'''

    total_seconds=(
        timestamp.dt.hour*3600+
        timestamp.dt.minute*60+
        timestamp.dt.second
    )

    day_in_seconds=24*60*60

    time_df=pd.DataFrame()

    time_df['time_sin']=np.sin(2*np.pi*total_seconds/day_in_seconds)
    time_df['time_cos']=np.cos(2*np.pi*total_seconds/day_in_seconds)

    return time_df



def prepare_aw_events_data(end_time:datetime|None=None) ->pd.DataFrame:
    '''get aw  event data and preprocess it , return a dataframe
        Args:
            end_time: the end time of fetching event
        
        Returns:
            pd.Dataframe:
                a dataframe contains following columns of data:

                 'title_category'(str): the combined series of aw category and title
                 'time_sin'(float):aw timestamp data after cyclical encoding
                 'time_cos'(float): another aw timestamp data after cyclical encoding
                 'duration'(float):aw duration data in seconds
                
            '''

    if end_time is None:
        end_time=datetime.now(tz=timezone.utc).astimezone()


    data=get_aw_raw_data(end_time=end_time,hours=24,hostname=HOSTNAME,port= PORT,host=HOST)

    if not data.empty:

        df=pd.DataFrame()

        df['title_category']=combine_aw_title_category(data)
        df['duration']=duration_transform(data['duration'])

        timestamp=convert_timezone(data['timestamp'])

        time_df=converted_aw_timestamp(timestamp)

        df['time_sin']=time_df['time_sin']
        
        df['time_cos']=time_df['time_cos']


        return df
    
    print(f"WARNING: AW data doesn't exist at end_time: {end_time}. Returned -100 as fallback")

    return -100
    


def pad_aw_sequence(aw_sequence:pd.Series,aw_tensor_feature_size:int) -> tuple[torch.Tensor,torch.Tensor]:
    '''Convert the aw_sequence(pd.Series) to a tensor and pad the aw sequence and its mask
        Args:
            aw_sequence (pd.Series): A pd.Series contain aw sequence data, 
                It would contain 
                either
                    torch.Tensor represent the valid sequence data itself:
                        Shape:(sequence_size,token_size)
                        Type: torch.Tensor
                or
                    -100 represent the missing sequence data

            aw_tensor_feature_size (int): The Tensor token size inside the sequence
            
                
        Returns:
            padded_data{torch.Tensor}:
                the aw_sequence input after padding
                Shape:(sequence_size,token_size)

            padded_mask{torch.Tensor}:
                the attention mask of padded_data
                Shape:(sequence_size,token_size)'''

    aw_data_list=[]
    mask_list=[]

    for t in aw_sequence:
        
        #Handle the -100 case
        if isinstance(t,torch.Tensor):

            aw_data_list.append(t)
            
            mask_list.append(torch.ones(t.shape[0],device=DEVICE)) # Valid token in the sequence

        else:

            aw_data_list.append(torch.zeros(size=(1,aw_tensor_feature_size),device=DEVICE))
            mask_list.append(torch.zeros(size=(1,),device=DEVICE))
                            
    
    padded_data=pad_sequence(aw_data_list,batch_first=True)
    
    padded_mask=pad_sequence(mask_list,batch_first=True) #After the padding, mask shape become (sequence_size,token_size)

    pprint.pprint('aw seuqnce shape: ', padded_data.shape)

    return padded_data,padded_mask


def prepare_sentence_transformer_input(aw_data:pd.DataFrame):
    '''preparet the sentence transformer's input
        Args:
            aw_data{pd.dataframe}: 
                Contain following column:
                    titel_category{str}: The string part of aw_data, contain a string of title and category
                    
                    time_sin{float}: The sin time after cyclic encoding
                    time_cos{float}: The cos time after cyclic encoding
                    duration{float}: The duration for each event in the aw data
                     
            
        Returns:
            aw_text{np.dnarray}: the string part of aw_data,
            
            num_tensor{torch.Tensor}: the numerical part of aw_data
            '''

    aw_text=aw_data['title_category'].values

    numerical_col=['time_sin','time_cos','duration']

    num_tensor=torch.tensor(aw_data[numerical_col].values,dtype=torch.float32).to(DEVICE)


    return aw_text,num_tensor


def encode_aw_data(aw_data:pd.DataFrame) -> dict:
    '''convert the aw_data to vector tensor
        Args:
            aw_data{pd.dataframe}: 
                Contain following column:
                    titel_category{str}: The string part of aw_data, contain a string of title and category
                    
                    time_sin{float}: The sin time after cyclic encoding
                    time_cos{float}: The cos time after cyclic encoding
                    duration{float}: The duration for each event in the aw data
                     
        Returns:

            A dic contain:
                A text tensor and a  numerical torch tensor.
                For more details, see Return string of forward function in sentence_transformer.py for text tensor
                and see the Rturn string of  prepare_sentence_transformer_input() in train_utils.py

            '''

    aw_text,num_tensor=prepare_sentence_transformer_input(aw_data)

    encoder=ActivityWatcherEncoder()

    
    text_tensor=encoder(aw_text=aw_text)
    
    tensor_dict={
        'aw_text_tensor':text_tensor.detach().cpu(),
        'aw_num_tensor':num_tensor.detach().cpu()
    }


    return tensor_dict


def convert_timestamp_to_tensor_series(timestamp:pd.Series) ->pd.Series:
    '''Convert the timestamp Series to  a tensor series
    
        Args:
            timstamp{pd.Series}: A pd.Series of ISO format timestamp

        Returns:
            A pd.Series of either the output of encode_aw_data function or -100.
            For more details, you could check the doc string of encode_aw_data()
            '''

    converted_timestamp=timestamp.apply(lambda onetime:convert_timezone(onetime) if onetime!=-100 and onetime!='-100' else -100)

    print('preparing AW events data...')
    #a pd.Series contain either dataframe or -100
    aw_data=converted_timestamp.apply(lambda onetime:prepare_aw_events_data(end_time=onetime) if onetime !=-100 else -100)

    print("Encoding AW events data..")
    aw_encode_data=aw_data.apply(lambda onedata:encode_aw_data(aw_data=onedata) if isinstance(onedata,pd.DataFrame) else -100)

    print("Data encoded!")
    return aw_encode_data








#----------------------------------------------------------------------------------------
#----------------------Helper function for train data processing-------------------------
#----------------------------------------------------------------------------------------



def data_visualize():
    '''vitualize a bit the duration'''

    data=pd.read_csv('ordered_output.csv',na_filter=False)

    total_second=pd.to_timedelta(data.loc[:,'duration'],errors='coerce').dt.total_seconds().fillna(0)


    #log transform for the better distribution
    log_second=np.log1p(total_second)
    
    mean_second=log_second.mean()
    std_second=log_second.std()

    z_score_second=(log_second-mean_second)/std_second

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    ax[0,0].hist(data['duration'])
    ax[0,0].set_title('original')
    ax[0,1].hist(total_second)
    ax[0,1].set_title('transform to second')
    ax[1,0].hist(log_second)
    ax[1,0].set_title('log transform')
    ax[1,1].hist(z_score_second)
    ax[1,1].set_title('z_score')

    plt.tight_layout()
    plt.show()






def prepare_log_training_data():
    '''prepare the training data for logistic model'''


    pd_like=pd.read_csv('liked_data.csv')
    pd_dis=pd.read_csv('disliked_data.csv',header=None,skiprows=1)

    pd_dis.columns=pd_like.columns

    
    #add one column that contain the ground truth 1 or 0
    pd_like['label']=1

    pd_dis['label']=0

    #combine two together
    combined_data=pd.concat([pd_like,pd_dis],ignore_index=True)

    randomized_data=combined_data.sample(frac=1).reset_index(drop=True)

    #select the feature 

    feature_col=['title', 'description','youtuber']

    x=randomized_data[feature_col]

    y=randomized_data['label']

    return x,y


def prepare_rf_data():
    '''prepare the training data for random forest'''

    video_data=pd.read_csv('labeled_data_1009_sorted.csv')

    x_col=['title','youtuber','description']

    x=copy.deepcopy(video_data[x_col].fillna(''))

    x.to_csv('test_x.csv',index=False)

    x['Productivity']=0
    x['Video']=0
    x['sin']=0
    x['cos']=0

    
    y=video_data['label']

    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=33,stratify=y)

    return train_x,test_x,train_y,test_y






def data_preprocess() ->None:
    '''Combine the history data and the labeled data from it and handel the missing part, convert the time to total second,fill the Nan with -100'''

    labelled_data=pd.read_csv('labeled_data_1009_sorted.csv')
    history_data=pd.read_csv('random_history_data_3000.csv')


    labelled_mask= history_data['videoId'].isin(labelled_data['videoId'])

    unlabelled_data=copy.deepcopy(history_data[~labelled_mask])

    unlabelled_data.loc[:,'tag']=np.nan
    unlabelled_data.loc[:,'interest']=0

    labelled_data.loc[:,'interest']=1

    #combine the data that is from history and unlabelled and the lebelled data
    whole_data=pd.concat([labelled_data,unlabelled_data],ignore_index=True)

    
    videoid_list=whole_data['videoId'].fillna('').tolist()
   
    cleaned_output=get_video_data(videoid_list,rm_stopwrords=False)

    #replace the text with the text that haven't removed the stopwords
    replace_col=['youtuber','description','title','data_state','date','duration']

    droped_ori_data=whole_data.drop(columns=replace_col)

    ordered_output=pd.merge(droped_ori_data,cleaned_output,how='left',on='videoId')

    #reorder the columns a bit
    order=['youtuber','title','description','videoId','data_state','date','duration','tag','interest','label']
    ordered_output=ordered_output[order].drop_duplicates(subset='videoId',keep='first')
    
    text_col=['youtuber','title','description']


    #NOTE be cautious about the code line below, if you get this kinda of error:
    # ValueError: text input must be of type `str` (single example), `list[str]` (batch or single pretokenized example) or `list[list[str]]` (batch of pretokenized
    #  replacing the line below with:
    # ordered_output[text_col]=ordered_output[text_col].replace(empyt_values,'').fillna('')
    # may help
    ordered_output[text_col]=ordered_output[text_col].fillna('')


    ordered_output.info()

    ordered_output.to_csv('ordered_output_with_hash.csv',index=False)



    

def like_dislike_streamlit_data_preprocess() ->None:
    '''concat the like, dislike data with tag labelled data '''

    db=PersonalFeedDatabase()

    like_data=db.get_data('like_data')
    dislike_data=db.get_data('dislike_data')
    streamlit_data=db.get_data('streamlit_data')

    like_data.loc[:,'interest']=1

    dislike_data.loc[:,'interest']=0


    total_df=pd.concat([like_data,dislike_data,streamlit_data],ignore_index=True).fillna('')

    db.save_data('interest_data',total_df)

    print('saved!')




def inerest_productive_data_preprocess()->None:
    '''preprocess the interest and productive model's training data'''

    like_dislike_streamlit_data_preprocess()
    db=PersonalFeedDatabase()

    interest=db.get_data('interest_data')

    productive_data=db.get_feedback()

    #mask videoId to productive rate

    productive_videoid=productive_data['videoId'].to_list()

    productive_df=get_video_data(productive_videoid,rm_stopwrords=False)

    print(productive_df)
    productive_df['interest']=-100
    productive_df['productive_rate']=productive_data['productive_rate'].fillna(-100)
    productive_df['timestamp']=productive_data['timestamp']

    print(productive_df)

    interest['productive_rate']=-100
    interest['timestamp']=-100
    interest=interest.rename(columns={'date':'upload_time'})
    print(interest)

    whole_data=pd.concat([interest,productive_df],ignore_index=True)

    #productive_df.to_csv(PROJECT_ROOT/'productive_data.csv')

    db.save_train_data(whole_data)
    print(whole_data)
    print('saved!')


def convert_timestamp_to_pt_file(timestamp:datetime|pd.Series,path:Path) ->None:
    '''convert the tiemstamp to pt file '''

    if isinstance(timestamp,datetime):
        timestamp=pd.Series(timestamp.isoformat())

    tensor_series=convert_timestamp_to_tensor_series(timestamp)

    print('tensor_series generated!')

    manifest_data={}

    for time,tensor_dict in zip(timestamp,tensor_series):

        safe_time=time.replace(':','-')
        tensor_file_name=f'{safe_time}.pt' #use the timstamp as name of the tensor


        torch.save(tensor_dict,path/tensor_file_name)
        print(f'{tensor_file_name} saved!')

        manifest_data[f'{time}']=tensor_file_name

    with open(path/'manifest.json','w',encoding='utf-8') as f:
        json.dump(manifest_data,f,indent=4) 

    print('Manifest saved!')



def timestamp_data_preprocess(path:Path) -> None:
    '''Convert each timestamp data to a Sentence Transformer encoded vector and save it into the path 
        Args:
            path(Path): The folder where you save the tensor data.'''
    db=PersonalFeedDatabase()
    
    productive_data=db.get_feedback().drop(columns='is_trained')

    timestamp=productive_data['timestamp']

    convert_timestamp_to_pt_file(timestamp,path)


    print('Converting timestamp to Tensor Series... ')

    


def manifest_process(path:Path) -> None:
    '''Helper function: use for motifying the manifest file to assume every feedback data in the database has a relate tensor file 
        '''

    db=PersonalFeedDatabase()
    
    productive_data=db.get_feedback().drop(columns='is_trained')

    timestamp=productive_data['timestamp']

    manifest_data={}

    for time in timestamp:

        safe_time=time.replace(':','-')
        tensor_file_name=f'{safe_time}.pt' #use the timstamp as name of the tensor

        manifest_data[f'{time}']=tensor_file_name

    with open(path/'manifest.json','w',encoding='utf-8') as f:
        json.dump(manifest_data,f,indent=4) 

    print('Manifest saved!')

    
    





    


    














