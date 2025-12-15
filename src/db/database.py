'''Contain the database class for whole project'''

import pprint
import sqlite3 as sq
import pandas as pd
import sqlalchemy

from enum import StrEnum
from datetime import datetime,timedelta

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import text

from typing import Literal,Any,Iterable,TypeAlias

from src.path import DATABASE_PATH


WriteTableName: TypeAlias = Literal[
    'like_data','dislike_data','interest_data'
    ]

ReadTableName: TypeAlias = Literal[
    'like_data', 'dislike_data', 'streamlit_data','train_data','history_data','interest_data'
    ]




class VrdndiDatabase:
    '''The database contain all training or content data for Vrdndi'''
    def __init__(self) -> None:

        self.dbpath=DATABASE_PATH/'vrdndi_db.sqlite'

        self.conn=sq.connect(self.dbpath)

        self.cursor=self.conn.cursor()

        self.engine=sqlalchemy.create_engine(f'sqlite:///{self.dbpath}')

        self._initial_registry_table()
        self._initial_streamlit_schema()
        self._initial_video_table()
        


    def _initial_registry_table(self) ->None:
        '''initial regisry table before saving data intot it'''

        with sq.connect(self.dbpath) as conn:
            query='''CREATE TABLE IF NOT EXISTS registry(
                name TEXT PRIMARY KEY,
                value INTEGER
            )
            '''

            conn.execute(query)


    def _set_registry_value(self,key:str,value:int) ->None:
        '''save a key value pair in registry table in the database'''

        with sq.connect(self.dbpath) as conn:

            conn.execute(
                "INSERT OR REPLACE INTO registry (name, value) VALUES (?,?)",
                (key,value)
            )

    def _get_registry_value(self,key:str) ->int:
        with sq.connect(self.dbpath) as conn:


            cursor=conn.execute("SELECT value FROM registry WHERE name = ?",(key,))

            row=cursor.fetchone()


        if not row:
            return 0
        
        return row[0]

    
    
    def _initial_video_table(self) ->None:
        '''Connect the vrdndi database'''
        with sq.connect(self.dbpath) as conn:
        

            conn.execute('''
            CREATE TABLE IF NOT EXISTS videos(
                        videoId TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        youtuber TEXT NOT NULL,
                        description TEXT,
                        upload_time TEXT,
                        duration TEXT,
                        data_state TEXT)

            ''')

        print('Created/ Connected to vrdndi database')

    def _insert_if_not_exits(self,table:Any,conn:sqlalchemy.engine.Connection,keys:list,data_iter:Iterable) ->int:
        '''Insert new data into target table '''

        data=[dict(zip(keys,row)) for row in data_iter]


        statement = insert(table.table).values(data)

        statement = statement.on_conflict_do_nothing(index_elements=['videoId'])

        rs=conn.execute(statement)

        return rs.rowcount
    
    def _initial_streamlit_schema(self) ->None:
        '''Make sure the data structure is correct before saving'''
        
        with self.engine.connect() as conn:
            query=text('CREATE UNIQUE INDEX IF NOT EXISTS idx_videoId ON streamlit_data("videoId")')

            conn.execute(query)


    def save_video_data(self,videodata:pd.DataFrame) ->None:
        '''Save the video data to personal_feed database'''
        
        videodata.to_sql(
            name='videos',
            con=self.conn,
            if_exists='append',
            index=False
        )

        self.conn.commit()

        self.conn.close()

        print('\n Video Data inserted from DataFrame successfully')

    def update_video_data(self,videodata:pd.DataFrame) ->None:
        '''Save the video data to personal_feed database'''

        
        videodata.to_sql(
            name='temp_data',
            con=self.conn,
            if_exists='replace',
            index=False
        )

        #NOTE I haven't implement the function that check if the column is matching or not
        query='''
        INSERT INTO videos
        SELECT *
        FROM temp_data
        WHERE NOT EXISTS (
        SELECT 1 FROM videos WHERE videos.videoId = temp_data.videoId
        )

        '''

        self.cursor.execute(query)
        self.cursor.execute(
            '''
        DROP TABLE temp_data
        '''
        )

        self.conn.commit()

        self.conn.close()

        print('\n Video Data inserted from DataFrame successfully')

    def _get_column_name(self) ->list:
        '''Get the table column name'''

        return [desc[0] for desc in self.cursor.description]


    def fetch_videos_from_past_days(self,fetch_range_days:int=3) -> pd.DataFrame:
        '''Fetch youtube videos from past 3 days(default) from your subscription
            Returns:
                A pd.DataFrame contains:
                    youtuber, description, title, videoId, data_state,duration, upload_time of each video '''


        now=datetime.now()

        threeday_ago=now-timedelta(days=fetch_range_days)

        query="SELECT * FROM videos WHERE upload_time >= ?"

        self.cursor.execute(query,(threeday_ago,))

        results=self.cursor.fetchall()

        column_name=self._get_column_name()

        

        df_data=pd.DataFrame(results,columns=column_name)


        return df_data
    

    def update_feed(self,data:pd.DataFrame):
        '''Store the current website feed in the database and replace the previous one'''

        data.to_sql(
            name='feed',
            con=self.conn,
            if_exists='replace',#it would drop entire table if exsits
            index=False
        )

        self.conn.commit()

        self.conn.close()

        print('\n Feed data updated successfully')


    def get_feed(self,order:Literal['interest','productive']='productive') ->pd.DataFrame:
        '''
        Get the feed data from database and order by decreasing productive_rate or interest
        depend on `order` argument (Highest on the top)
        
        Args:
            order: The order of feed
                - interest: Get the feed sorted by interest rate descent
                - productive: Get the feed sorted by productive rate descent
        
        '''

        with sq.connect(self.dbpath) as conn:

            if order =='productive':
                feed=pd.read_sql_query("SELECT * FROM feed ORDER BY productive_rate DESC",conn)
            else:
                feed=pd.read_sql_query("SELECT * FROM feed ORDER BY interest DESC",conn)

            

        return feed
    
    def save_feedback(self,data:pd.DataFrame,if_exists:Literal['append','fail','replace']='append'):
        '''Save the feedback to the database'''

        with sq.connect(self.dbpath) as conn:

            data.to_sql(
                name='feedback',
                con=conn,
                if_exists=if_exists,
                index=False
            )


        print('\n Feedback data updated successfully')

    def get_feedback(self) ->pd.DataFrame:
        '''Get the feedback data from database'''

        with sq.connect(self.dbpath) as conn:
            feedback=pd.read_sql_query("SELECT * FROM feedback WHERE is_trained = 0",conn)

        
        return feedback
    
    def save_train_data(self,data:pd.DataFrame):
        '''Save the train data to database'''


        with sq.connect(self.dbpath) as conn:

            data.to_sql(
                name='train_data',
                con=conn,
                if_exists='replace', #Replace the current data
                index=False

            )
        print('\n Train data updated successfully')

    def update_feed_state(self,value:Literal[0,1]) ->None:
        '''Update feed state to either 0 or 1 '''

        self._set_registry_value('feed_state',value=value)


    def get_feed_state(self) ->bool:
        '''Get the feed state from database'''
        

        rs=self._get_registry_value('feed_state')

        if rs==1:
            return True

        return False


    def save_history_data(self,history_data:pd.DataFrame) ->None:
        '''
        Save the history data into database. 
        '''


        with sq.connect(self.dbpath) as conn:


            history_data.to_sql(
                name='history_data',
                con=conn,
                if_exists='replace', 
                index=False

            )

        print('History data saved into database!')





    def save_streamlit_data(self,data:pd.DataFrame) ->None:
        '''Save the streamlit labeled data to database'''
        pprint.pprint(data)



        data.to_sql('streamlit_data',con=self.engine,if_exists='append',index=False,method=self._insert_if_not_exits)


        print('Sucessfully saved streamlit data to database')

    

    def update_streamlit_index(self,index:int) ->None:
        '''Save the index of last datapoint into database'''


        self._set_registry_value('streamlit_index',index)

        print(f'Set streamlit index to {index} ')


    def get_streamlit_index(self) ->int:
        '''Get the feed state from database'''

        index=self._get_registry_value('streamlit_index')


        return index
    

    def save_data(self,table_name:WriteTableName,data:pd.DataFrame) ->None:
        '''Save the like dislike data into database
        Note:
            If the data already existed, it would fail (if_exists = 'fail')'''

        with sq.connect(self.dbpath) as conn:
            data.to_sql(f'{table_name}',con=conn)


    def get_data(self,table_name:ReadTableName) ->pd.DataFrame:
        '''Get the data from database via inputing its table name
        '''

        with sq.connect(self.dbpath) as conn:
            query=f"SELECT * FROM {table_name}"

            data=pd.read_sql(query,conn)

        return data
    
    






    
    


        




        
            



        



        




        

        


            


        



        





