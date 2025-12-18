'''this file contain the data fetching and cleaning part'''
import logging
import re
import os
import pickle
import time
import pprint
import socket
import numpy as np
import pandas as pd
import jieba
import copy
import googleapiclient.discovery
import requests
import torch
import pytz
import regex

from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pad_sequence



from matplotlib import pyplot as plt

from datetime import datetime, timedelta,timezone

from aw_client import ActivityWatchClient
from aw_client.queries import canonicalEvents,DesktopQueryParams
from aw_client.classes import get_classes

from nltk.corpus import stopwords

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

from src.models.activity_watcher_encoder import ActivityWatchEncoder
from src.db.database import VrdndiDatabase
from src.config import CLIENT_SECRET_FILE

from src.path import SECRET_PATH,ASSETS_PATH,RAW_DATA_PATH

db=VrdndiDatabase()

#------------------------------PART ONE: DATA COLLECTING---------------------------------------


def get_auth_ser():
    '''authenticate via oauth2, return the credential'''

    credentials=None
    tokenfile='token.pickle'
    secret_file=CLIENT_SECRET_FILE

    if os.path.exists(SECRET_PATH/tokenfile):
        with open(SECRET_PATH/tokenfile,'rb') as token :
            credentials=pickle.load(token)

    #if there's no credential, got one
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

        else:
            flow=InstalledAppFlow.from_client_secrets_file(
                SECRET_PATH/secret_file,
                scopes= ['https://www.googleapis.com/auth/youtube.readonly'])
            credentials=flow.run_local_server(port=0)

        with open(SECRET_PATH/'token.pickle','wb') as token:
            pickle.dump(credentials,token)

    return credentials

def fetch_sub_youtuber(credential):
    '''fetch the youtuber's data in user's subscription from youtube via oauth2'''

    youtube=googleapiclient.discovery.build('youtube','v3',credentials=credential)

    all_sub=[]
    next_page_token=None


    #get all the subscription vai look for pages
    while True:

        requests=youtube.subscriptions().list(
            part="snippet,contentDetails",
            mine=True,
            maxResults=50,
            pageToken=next_page_token
        )

        response=requests.execute()
        
        all_sub.append(response)

        next_page_token=response.get('nextPageToken')

        if not next_page_token:
            break

    
    
    return all_sub

def get_sub_video(yter_data, credential ,sub_video_num:int=10):
    '''fetch the new video in my subscription from the subscribed youtuber via oauth2'''
    youtube=googleapiclient.discovery.build('youtube','v3',credentials=credential)

    sub_video_list=[]
    for sub in yter_data:
        request=youtube.activities().list(
            part="snippet,contentDetails",
            maxResults=sub_video_num,
            channelId=sub['channelId']
        )
        video_data = request.execute()
        sub_video_list.append(video_data)


    return sub_video_list

def get_liked_video(credential,dislike=False):
    '''fetch all the video that is liked'''
    
    youtube=googleapiclient.discovery.build('youtube','v3',credentials=credential)

    next_page_token=None
    all_liked_v=[]

    n=0
    while True:
        
        request=youtube.videos().list(
            part="snippet,contentDetails",
            myRating="like" if not dislike else 'dislike',
            maxResults=50,
            pageToken=next_page_token
        )
        response=request.execute()
        
        all_liked_v.append(response)

        next_page_token=response.get('nextPageToken')
        
        n+=1
        if not next_page_token:
            break

        if n>100:
            break
    
    return all_liked_v

def get_video(videoid) ->list:
    '''get the video via videoid'''

    cred=get_auth_ser()
    youtuber=googleapiclient.discovery.build('youtube','v3',credentials=cred)
    
    all_video=[]
    
    if isinstance(videoid,str):

        request=youtuber.videos().list(
            part='snippet,contentDetails',
            id=videoid,
            maxResults=50,
        )
        
        response=request.execute()
        
        return response

    #for the case the videoid is a list
    for sub_id in videoid:
        
        if not sub_id:
            continue

        request=youtuber.videos().list(
            part='snippet,contentDetails',
            id=sub_id,
            maxResults=50,
        )
        
        response=request.execute()

        time.sleep(0.5)
        
        all_video.append(response)
        
        
    return all_video



def load_history_data(sampel_n:int|None=None) ->list:
    '''Load the history data and return video id
        Args:
            sampel_n(int): The number of videos fetch from history data. Default to None, return all videodata

        Returns:
            list contain list of chunks of videoId. Each chunk has 50 videoIds
    
    '''
    history_data=pd.read_json(RAW_DATA_PATH/'watch-history.json')
    
    #Remove the watched history of ad
    mask_to_keep=history_data['details'].isna()
    history_data_no_ad=history_data.loc[mask_to_keep]
    
    mask_to_remove= history_data_no_ad['title'].str.contains('short')

    no_short_data=history_data_no_ad.loc[~mask_to_remove]

    #Remove post or other stuff that don't have a id
    video_mask=no_short_data['titleUrl'].str.contains('\u003d',na=False)
    video_only_data=no_short_data.loc[video_mask]
    
    #If sample_n is None,then use all the video data
    if sampel_n:
        video_df=video_only_data.sample(n=sampel_n)
    else:
        video_df=video_only_data

    splited_list=video_df['titleUrl'].str.split('\u003d').to_list()
    
    #It would be a list of list, inner list contain two part of url, latter is video id
    videoid=[url[1] for url in splited_list]

    
    if len(videoid)>50:
        id_list=videoid_split(videoid)
    else:
        id_list=[','.join(videoid)]

    
    return id_list


def get_aw_raw_data(*,end_time:datetime|None=None,hours:int=3,hostname:str|None=None,
                    host:str|None = None,port:int|None = None
                    ) ->pd.DataFrame:
    '''fetch the aw raw data. 
    If you are not fetching the data from another computer, no need for inputing host and port

    Args:
        end_time{datetime}: Fetch the aw data before this time. 
            Defualt to None. In that case, end_time would be now (current time)
        hours{int}: How many hours of data you want to fetch, default to 3 hours
        hostname{str}: The network name of the device ( You may find it in the dashboard of ActivityWatch)
            Default to None. In that case, hostname would be your current deivce's automatically
        host{str}: The server's IPv4 address, default to None,
            which mean you would fetch the data from your current device
            (e.g. '100.100.x.x' for using Tailscale , for localhost, just leave it to be default)

        port{str}: The port number where the ActivityWatch's server is listening to. 
            Default to None, leave it to be handled by aw_client


    Returns:
        A normalized pd.DataFrame contains following columns:
            ['duration', 'id', 'timestamp', 'data.$category', 'data.app', 'data.title', 'data.url']
         
        For more details, please see the examples below

    Examples:
        >>> #Fetch from another computer
        >>> print(get_aw_raw_data(hostname='someoneMacBook-Air.local',host='100.100.66.42',port=5600)) 
        
        >>> | duration |     id | timestamp           | App | Category          | Title                           | url |
            |---------:|-------:|:--------------------|:----|:------------------|:--------------------------------|:----|
            |   218.9s | 124794 | 2025-11-22 03:31:46 | Zen | [Uncategorized]   | Baldur's Gate 3 !               | NaN |
            |   193.6s | 124796 | 2025-11-22 03:35:25 | Zen | [Uncategorized]   | Best game of the world!         | NaN |
            |   749.5s | 124798 | 2025-11-22 03:38:39 | Zen | [Media, Video]    | Youtube: Something interest     | NaN |
            |     1.1s | 124801 | 2025-11-22 03:51:08 | Zen | [Media, Video]    | Bilibili: Short Clip...         | NaN |
            |     1.0s | 124803 | 2025-11-22 03:51:09 | Zen | [Uncategorized]   | Zen Browser New Tab             | NaN |

    '''
    if end_time is None:

        end_time=datetime.now(tz=timezone.utc).astimezone()
    
    
    to_hour=timedelta(hours=hours)

    aw=ActivityWatchClient(host=host,port= port)

    web_class=get_classes()
    #if you haven't set the class in the aw dashboard, it may return a 'None' error 
    #(i.e. It doesn't return default classes as fallback somehow)
    #In that case, you may wanna read the code in aw_client(Github project)/aw_client(Folder)/classes.py/get_classes(Function)
    #Or simply set the category(classes) in the ActivityWatch's dashboard (If you haven't set it yet! )

    if hostname is None:
        hostname=socket.gethostname()

    canonicalquery = canonicalEvents(
        DesktopQueryParams(
            bid_window=f"aw-watcher-window_{hostname}",
            bid_afk=f"aw-watcher-afk_{hostname}",
            classes=web_class,
        )
    )


    query= f"""{canonicalquery} RETURN = events;"""

    #Try to get the raw data

    data=aw.query(query,[(end_time-to_hour,end_time)])

    #Normalize data,tranform the data.$category to a list like form
    nor_data=pd.json_normalize(data[0])

    return nor_data



def videoid_split(videoid_list) ->list:
    '''split the video id and group the videoId to a chunk of 50'''

    #make sub list, each has 50 video id
    splited_list=[videoid_list[i:i+50] for i in range(0,len(videoid_list),50)]

    #prepare input for youtube api
    input_list=[','.join(sub_list) for sub_list in splited_list]

    return input_list


def get_video_data(videoid:list,rm_stopwrords=True) ->pd.DataFrame:
    '''get the video via api by using video id list
    Args:
        videoid: a list of video id, each element in the list should be a single video id
        '''
    
    videoid_list=videoid_split(videoid)

    video_list=clean_yt_video(get_video(videoid_list),rm_stopwords=rm_stopwrords)

    df=pd.DataFrame(video_list)

    return df




#------------------------------PART TWO: DATA CLEANING---------------------------------------




def clean_sub_data(data):
    '''get the channel id,name,description from subscribed youtuber'''

    
    cleaned_data=[]
    for page in data:
        subscription_list=page['items']

        for sub in subscription_list:

            cleaned_sub={
                'youtuber':clean_yt_data_for_training(sub['snippet']['title'],True),
                'channelId':sub['snippet']['resourceId']['channelId'],
                'description':clean_yt_data_for_training(sub['snippet']['description'],True),

            }
            cleaned_data.append(cleaned_sub)

    return cleaned_data


def is_valid_token(text):
    '''check if a token is english or chinese, if so return True, otherwise False'''

    is_english=text.isalpha()
    
    is_chinese=all('\u4400'<=char<='\u9fff' for char in jieba.cut(text))

    return is_english or is_chinese


def load_cn_stop_words():
    '''load the Chinese stop words '''
    stop_words=set()

    
    with open(ASSETS_PATH/'hit_stopwords.txt','r',encoding='utf-8') as f:
        for line in f:
            stop_words.add(line.strip())

    return stop_words


def detect_lang(text):
    '''detect if the text belong to english or chinese, otherwise False'''
    
    if re.search(r'[\u4e00-\u9fff]',text):
        return 'cn'
    
    return 'other'


def remove_stop_words(text):
    '''remove the stop wrods from the text'''
    
    en_stop_words=set(stopwords.words('English'))

    cn_stop_word=load_cn_stop_words()

    removed_stopword_text=[word for word in text if word not in cn_stop_word and word not in en_stop_words ]

    return ' '.join(removed_stopword_text)


def mix_lang_tokenise(text):
    '''keep english and chinese word and %, separete them'''

    
    #segments=re.findall(r'[\u4e00-\u9fff]+|\d+(?:%|#)|#[a-zA-Z0-9]+|[a-zA-Z0-9]+',text)

    #keep all the word, number and punctuation in all language
    segments=regex.findall(r'[\p{L}\p{N}]+|\p{P}',text)

    final_tokens=[]
    
    for segment in segments:
        if detect_lang(segment)=='cn':
            final_tokens.extend(list(jieba.cut(segment)))
        else:
            final_tokens.append(segment.lower())

    return final_tokens

def clean_video_data(video,rm_stopwords=True):
    '''clean single video data'''

    if  'youtube#activity' not in video['kind'] :
        cleaned_video={
            'youtuber':clean_yt_data_for_training(video['snippet'].get('channelTitle','')),
            'description':clean_yt_data_for_training(video['snippet'].get('description',''),rm_stopwords),
            'title':clean_yt_data_for_training(video['snippet'].get('title',''),rm_stopwords),
            'videoId':video['id'] ,
            'data_state':video['kind'],
            'duration':video['contentDetails'].get('duration',''),
            'upload_time':video['snippet']['publishedAt'],
        }
    else:
        #NOTE there's no duration in such format
        cleaned_video={
                'youtuber':clean_yt_data_for_training(video['snippet'].get('channelTitle','')),
                'description':clean_yt_data_for_training(video['snippet'].get('description',''),rm_stopwords),
                'title':clean_yt_data_for_training(video['snippet'].get('title',''),rm_stopwords),
                'videoId':video['contentDetails']['upload']['videoId'] if video['snippet']['type']=='upload' else None,
                'data_state':video['snippet']['type'],
                'upload_time':video['snippet']['publishedAt'],
            }
    
    
    return cleaned_video



#switched the default from False to True, haven't test the effect
def clean_yt_data_for_training(text,rm_stopword=True): 
    '''clean the data for training'''
    #NOTE you haven't implement the bolierplate removing function
    
    url_removed_text=re.sub(r'https?://\S+|www\.\S+','',text)
    
    space_striped_text=re.sub(r'\s+',' ',url_removed_text)
    preprocess_text=mix_lang_tokenise(space_striped_text)

    if rm_stopword:
        rm_stopword_text=remove_stop_words(preprocess_text)


    else:
        rm_stopword_text=' '.join(preprocess_text)



    return rm_stopword_text

def clean_liked_video(video_data,rm_stopwords:bool=True):
    '''clean the liked video'''

    cleaned_data=[]
    #for each page
    for page in video_data:
        items=page['items']
        for video in items:
            
            cleaned_video=clean_video_data(video,rm_stopwords=rm_stopwords)

            cleaned_data.append(cleaned_video)

    return cleaned_data



def clean_sub_video_data(video_data,rm_stopwords:bool = True) -> pd.DataFrame:
    '''clean the new video data from subscribed youtuber'''
    
    cleaned_data=[]

    #load the dic of each subscribed youtuber from list
    for yter in video_data:
        each_yter=yter['items']
        
        #for each content/video of the youtuber
        for video in each_yter:
            
            cleaned_video=clean_video_data(video,rm_stopwords)

            cleaned_data.append(cleaned_video)
    
    df=pd.DataFrame(cleaned_data)

    filtered_mak=df['data_state']=='upload'

    filtered_df=df[filtered_mak]


    return filtered_df



def clean_yt_video(data_list,rm_stopwords=True):
    '''get and clean video by default '''

    
    if isinstance(data_list,dict):
        video_list=data_list['items']

        #handel the exception, find no video
        if video_list==[]:
            return None

        cleaned_video_list=[]
        for video in video_list:
            cleaned_video_list.append(clean_video_data(video,rm_stopwords))
        
        return cleaned_video_list
            
    if isinstance(data_list,list):

        cleaned_video_list=[]

        for page in data_list:
            
            video_list=page['items']

            for video in video_list:
                
                #somehow the items in video would be list 
                cleaned_video_list.append(clean_video_data(video,rm_stopwords))

            
        return cleaned_video_list
    
    return None



def get_and_clean_yt_video_data(sub_video_num:int =10 ,rm_stopwords:bool = True) -> pd.DataFrame:
    '''gather fetching and cleaning function of youtube data to get the sub video data
        Args:
            sub_video_num: fetch how many video from each subscribed youtuber 
            rm_stopswords: whether or not remove stopwrods from text, default True'''

    credential=get_auth_ser()

    logging.info('Fetching subscribed youtuber...')
    sub_data=fetch_sub_youtuber(credential)

    logging.info('Cleaning youtuber data....')
    cleaned_sub_data=clean_sub_data(sub_data)

    logging.info('Fetching video from your subscribed youtuber.....')
    video_data=get_sub_video(cleaned_sub_data,credential,sub_video_num)

    logging.info('Cleaning video data.....')
    cleaned_video_data=clean_sub_video_data(video_data,rm_stopwords)

    #NOTE you may wanna return the sub data lately

    return cleaned_video_data



def get_and_clean_his_video_data(sample_n:int|None=None,rm_stopwords:bool=True) ->pd.DataFrame:
    '''randomly fetch video from history and clean the data, return a dataframe'''

    videoid=load_history_data(sample_n)
    data_list=get_video(videoid)

    cleaned_list=clean_yt_video(data_list,rm_stopwords=rm_stopwords)

    return pd.DataFrame(cleaned_list) 



def get_and_clean_liked_data(dislike:bool=False,rm_stopwords:bool=True) ->list:
    '''get cleaned liked video data'''

    return clean_liked_video(get_liked_video(get_auth_ser(),dislike),rm_stopwords=rm_stopwords)


def get_and_save_liked_disliked_data_for_database()->None:
    '''save the liked and disliked data from youtube to your database '''
    liked_data=get_and_clean_liked_data(rm_stopwords=False)

    disliked_data=get_and_clean_liked_data(dislike=True,rm_stopwords=False)

    df_liked=pd.DataFrame(liked_data)
    df_disliked=pd.DataFrame(disliked_data)
    
    db.save_data('like_data',data=df_liked)

    db.save_data('dislike_data',data=df_disliked)



def get_and_save_his_data_for_database(video_n:int|None=None, shorts:bool=True) ->None:
    ''' Randomly fetch the video in history, remove shorts, save to database
        Args:
            video_n (int,optional): The number of vide that fetch from history.
                For detail, check te args part in function load_history_data.
            shorts (bool,optional): Whether remove the shorts video (less than 60 seconds) from the history data. Default to True
            
            '''

    video_df=get_and_clean_his_video_data(video_n,rm_stopwords=False)

    video_df=video_df.drop_duplicates('videoId',keep='first')
    
    #Filter the video that is less than 60 seconds
    if shorts:
        has_hour=video_df['duration'].str.contains('H')
        has_minute=video_df['duration'].str.contains('M')

        no_short_mask= has_hour|has_minute

        no_short_video=video_df[no_short_mask]

        db.save_history_data(no_short_video)

        return
    
    db.save_history_data(video_df)
    


def get_and_save_yt_video_for_database() ->None:
    '''get the youtube subscription video data'''

    logging.info('Fetching Video (It may take a long time) ....')

    #Use a really large sub_video_num to fetch as many as possible from each subscribed youtuber
    
    videodata=get_and_clean_yt_video_data(sub_video_num=100000,rm_stopwords=False)

    video_id=videodata['videoId'].to_list()
    #And the video data fetch from channel won't have duration information of each video
    #So we fetch again by using its videoId
    real_videodata=get_video_data(video_id,rm_stopwrords=False)

    renamed_videodata=real_videodata.rename(columns={
        'date':'upload_time'
    })


    db.save_video_data(renamed_videodata)


def convert_timezone(timestamp:pd.Series|str,
                     timezone_name:str ='Asia/Hong_Kong',
                     utc:bool=False
                     ) ->pd.Series|datetime:
    '''convert the timestamp to Hongkong timezone'''



    converted_timestamp=pd.to_datetime(timestamp,format='ISO8601',utc=True)

    timezone=pytz.timezone(timezone_name)

    utc_tz=pytz.timezone('UTC')
    if isinstance(timestamp,str):
        curr_timestamp=converted_timestamp.astimezone(timezone)
    else:
        
        curr_timestamp=converted_timestamp.dt.tz_convert(timezone if not utc else utc_tz)


    return curr_timestamp





def get_aw_duration() -> pd.Series:
    '''get the duration of each catergory
    '''
    
    data=get_aw_raw_data()

    #create a new row for each category 
    data=data.explode('data.$category')

    duration=data.groupby(['data.$category'])['duration'].sum()

    return duration



def delete_aw_priavte():
    ''' delete all the data that contain Priavte in the title
        
        Notes:
            This function is from aw forum.
            You could find this function at https://forum.activitywatch.net/t/how-to-delete-data/557
            '''

    hostname=socket.gethostname()
    end_time=datetime.now(timezone.utc)
    start_time=end_time-timedelta(days=30)


    bucket = f"aw-watcher-window_{hostname}"
    pool = ThreadPoolExecutor(max_workers=10)
    contains = "Private"
    events = requests.get(f'http://localhost:5600/api/0/buckets/{bucket}/events?starttime={start_time.isoformat()}&endtime={end_time.isoformat()}').json()



    def delete(e):
        resp =  requests.delete(f'http://localhost:5600/api/0/buckets/{bucket}/events/{e["id"]}')
        if resp.status_code != 200:
            logging.error(f"failed to delete event  { e['data']['title']}")
        else:
            logging.info(f"deleted event {e['data']['title']}" )


    for e in events:
        
        if contains in e['data']['title']:

            logging.info(e['data']['title'])

            pool.submit(delete,e)



def duration_transform(duration:pd.Series) ->pd.Series:
    '''Apply log transform and Z-score to numerical duration
        Args:
            duration{duration}: A pd.Series that contain numerical duration'''
    
    #log transform for the better distribution, it's still pd.Series
    log_second=np.log1p(duration)
    
    mean_second=log_second.mean()
    std_second=log_second.std()

    z_score_second=(log_second-mean_second)/std_second

    return z_score_second
    


def iso_duration_transform(duration:pd.Series) -> pd.Series:
    '''Transform ISO 8601 duration to number and apply log tranformation and z score
        Args:
            duration{pd.Series}: 
                A pd.Series that contain ISO 8601 Duration (e.g. PT1H15M30S).'''
    
    #convert the video string duration to numerical duration in seconds
    total_second=pd.to_timedelta(duration,errors='coerce').dt.total_seconds().fillna(0)


    z_score_second=duration_transform(total_second)



    return z_score_second



#---------------------FEED DATA PREPARE----------------------------


def prepare_log_model_feed() ->pd.DataFrame:
    '''prepare the data that use to predict by logsitic regression'''

    raw_feed=pd.read_csv('feed.csv')

    feature_col=['title', 'description','youtuber']

    x=raw_feed[feature_col].fillna('')

    return x,raw_feed

def prepare_rf_model_feed(data):
    '''prepare the data that use to predict by random forest'''


    feature_col=['title', 'description','youtuber']

    x=data[feature_col].fillna('')

    duration=get_aw_duration()

    
    
    x['Productivity']=duration['Productivity']

    x['Video']=duration['Video']
    
    x['sin']=0
    x['cos']=0

    return x

def prepare_mt_pred_data(sub_video_num:int = 10) -> None:
        '''prepare the data for multi task model to predict'''

        #fetching from subcriptioin youtuber doesn't have duration
        data=get_and_clean_yt_video_data(sub_video_num=sub_video_num,rm_stopwords=False) 

        videoid_list=data['videoId']

        duration_contained_data=clean_yt_video(get_video(videoid_list),rm_stopwords=False)

        df_data=pd.DataFrame(duration_contained_data)

        current_time=datetime.now()
        df_data.to_csv(f'MTmodel_feed_{current_time}.csv',index=False)
        logging.info('feed data is exported!')

