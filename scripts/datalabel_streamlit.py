'''this file contain the streamlit function for labeling data'''


import pandas as pd
import streamlit as st
from typing import Literal


from src.db.database import VrdndiDatabase

@st.cache_resource
def get_database():
    '''get database'''

    return VrdndiDatabase()




class StreamlitDataLabel:
    '''
    Use streamlit to label data

    It would create a website to label interest data,
      and include some other feature to show the data state
    
    
    '''
    def __init__(self) -> None:
        self.db=get_database()


        
        
    def _setup_sidebar(self) ->None:
        '''Setup the sidebar in the streamlit page'''

        st.sidebar.markdown('video index')
        st.sidebar.write(st.session_state.video_index)

        st.sidebar.markdown('labeled_video_num_in_list')
        st.sidebar.write(len(st.session_state.labeled_video_list))


    def _initial_session_state(self) ->None:
        '''
        Initial some list or parameter in the session state to save video index,
        labeled video 
        '''
        #ready to count the datat number inside the label


        if 'video_index' not in st.session_state:
            st.session_state.video_index=0

        if 'labeled_video_list' not in st.session_state:
            st.session_state.labeled_video_list=[]


        if len(st.session_state.labeled_video_list)!=0:

            if 'label' in st.session_state.labeled_video_list[-1]:

                label=pd.DataFrame(st.session_state.labeled_video_list)['label']

                st.sidebar.markdown('Label')
                st.sidebar.write(label.value_counts())

            st.sidebar.markdown('last_elem_of_list')
            st.sidebar.write(st.session_state.labeled_video_list[-1]['title'])



    def _label_video(self,video:pd.Series,videoid:str,value:Literal[0,1]) ->None:
        '''label the video with 0 or 1'''

        assert isinstance(video,pd.Series),'arg video should be a pd.Series'

        video['interest']=value

        st.session_state.labeled_video_list.append(video)

        st.session_state.video_index+=1

        print(f'labeled video {videoid}')


    def _skip_video(self,videoid:str) ->None:
        '''skip current video, move on to next video'''

        st.session_state.video_index+=1
        print(f'Skiped video: {videoid} !')



    def _save_labeled_data(self) ->None:
        '''save labeled data to database'''

        df_labeled_data=pd.DataFrame(st.session_state.labeled_video_list)
        
        self.db.save_streamlit_data(df_labeled_data)
        self.db.update_streamlit_index(st.session_state.video_index)

        print('Saved labeled data to database!')

    def _move_to_previous_video(self) ->None:
        '''move back to previous video'''
        
        if st.session_state.video_index>0:
            st.session_state.video_index-=1


    
    def _undo_last_labeled_video(self) ->None:
        '''remove the last labeled video from the list'''

        del st.session_state.labeled_video_list[-1]




    def _load_data(self) ->None:
        '''load data from database to continue labeling'''


        index=self.db.get_streamlit_index()

        st.session_state.video_index=index






    def start_to_lable_data(self) ->None:
        '''label the data via streamlit'''

        video_data=self.db.get_data('history_data')


        self._initial_session_state()
        self._setup_sidebar()

        
        video_id=video_data['videoId'][st.session_state.video_index]
        video=video_data.iloc[st.session_state.video_index]

        url=f"https://www.youtube.com/watch?v={video_id}"

        st.set_page_config(layout='wide')
        col3,col4,col5,col6,col7=st.columns(5)
        st.video(url)
        col1,col2=st.columns(2)


        with col1:
            st.button('Interest',on_click=self._label_video,args=(video,video_id,1))
               
                    
        with col2:
            st.button('Uninterest',on_click=self._label_video,args=(video,video_id,0))
                

        with col3:
            
            st.button('Skip',on_click=self._skip_video,args=(video_id,))

            
        with col4:
            st.button('Save',on_click=self._save_labeled_data)

        with col5:
        
            st.button('Previous',on_click=self._move_to_previous_video)


        with col6:
            st.button('Undo',on_click=self._undo_last_labeled_video)


        with col7:
            st.button('Load data',on_click=self._load_data)


        

if __name__=="__main__":
    sdl=StreamlitDataLabel()
    sdl.start_to_lable_data()