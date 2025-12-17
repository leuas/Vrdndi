'''this is the frontend of the Vrdndi webstie'''
import cryptography
import numpy as np
import pandas as pd
import os
import logging
import nicegui
import time

import functools
import inspect

from functools import partial

from zoneinfo import ZoneInfo
from pathlib import Path
from datetime import datetime,timezone
from nicegui import ui,app
from src.config import ENABLE_EXPERIMENTAL_FEATURES,WEBSITE_HOST

from src.db.database import VrdndiDatabase


db=VrdndiDatabase()

new_feed_state={
        'feed_state':db.get_feed_state()
    }

class UpdateWebsitePage:
    '''The class for switching website page between feed main page and video player'''

    def __init__(self,video_play_container,feed_container) -> None:
        self.video_play_container=video_play_container

        self.feed_container=feed_container


    async def show_feed(self):
        '''show the feed in the main website'''

        self.video_play_container.clear()

        self.feed_container.visible=True
        self.video_play_container.visible=False

        scroll_pos=app.storage.general.get('feed_scroll_pos',0)
        #load the last position user scrolled to
        if scroll_pos>0:
            ui.timer(0.1,lambda:ui.run_javascript(f'window.scrollTo(0,{scroll_pos})'),once=True)


    async def show_video_player(self,videoid):
        '''show video palyer in the main website'''

        try:
            scroll_y=await ui.run_javascript('window.scrollY')
            app.storage.general['feed_scroll_pos']=scroll_y

        except Exception as e:
            logging.error(f'Could not get scroll position: {e}')

        watch_state={
        'start_time':None,
        'duration':[],
        'videoid':videoid
        }

        
        
        with self.video_play_container:

            #video_frame=f'''<iframe width="1120" height="630" src="https://www.youtube.com/embed/{videoid}" frameborder="0" allowfullscreen></iframe>'''

            video_frame=f'''<iframe width="1120" height="630" src="https://www.youtube.com/embed/{videoid}" frameborder="0" allowfullscreen></iframe>'''
            ui.html(video_frame,sanitize=False).classes('w-full mx-auto')

            
            
            with ui.row().classes('mx-auto'):
                with ui.button_group():
                    ui.button('interest',on_click=lambda:save_feedback(videoid,'interesting',1))
                    ui.button('uninterest',on_click=lambda:save_feedback(videoid,'interesting',0))
                
                with ui.button_group():
                    ui.button('correct timing',on_click=lambda:save_feedback(videoid,'productive_rate',1))
                    ui.button('not now',on_click=lambda:save_feedback(videoid,'productive_rate',0))


            ui.button('Back to feed').classes('text-left block mt-4').on_click(self.show_feed)


        self.feed_container.visible=False
        self.video_play_container.visible=True





def refresh_website_for_new_feed():
    '''update the feed if user click the button'''

    if db.get_feed_state is True:

        
        new_feed_state['feed_state']=False

        db.update_feed_state(0) #reset the feed_state value
        logging.info(db.get_feed_state())
        render_feed.refresh()

    else:
        ui.notification('You have already updated feed')
        new_feed_state['feed_state']=False
        db.update_feed_state(0) #reset the feed_state value
    
        logging.info(db.get_feed_state())

@app.post('/trigger-update')
def webhook():
    '''update the button visibility whitout reloading the website'''

    new_feed_state['feed_state']=True
    
def new_feed_arrive():
    '''Have new feed in the database'''
    db.update_feed_state(1)

    logging.info('feed arrived')




def save_single_feedback(value:int,key:str,videoid:str) ->pd.DataFrame:
    '''save the feedback'''

    hk_tz=ZoneInfo('Asia/Hong_Kong')

    now=datetime.now(hk_tz)

    now= now.replace(tzinfo=None) #just in case some app would convert it back to UTC

    feedback={
                'videoId': videoid,
                'interesting':np.nan,
                'productive_rate':np.nan,
                'timestamp':now.isoformat() if key=='productive_rate' else -100,
                'is_trained':0
            }
    df_feedback=pd.DataFrame([feedback])

    df_feedback[key]=value


    return df_feedback




def save_feedback(videoid,key,feedback_value) -> None:
    '''Save feedback to database'''
    
    new_feedback=save_single_feedback(feedback_value,key,videoid)
        
    db.save_feedback(new_feedback)

    ui.notification('feedback saved!')




@ui.refreshable
def render_feed(update_page:UpdateWebsitePage,video_per_load):
    '''load the feed to display'''
    data=db.get_feed()

    titles=data['title']
    video_ids=data['videoId']
    

    context_container=ui.grid(columns=3).classes('mx-auto')

    load_more_button=ui.button('load more').classes('text-left block mt-4')

    async def load_more() ->None:
        '''load the next batch of videos'''

        start=app.storage.general['videos_shown']

        end=start+video_per_load

        new_videos=video_ids[start:end]
        new_titles=titles[start:end]

        
        with context_container:

            for vid,title in zip(new_videos,new_titles):


                thumbnail=f'https://img.youtube.com/vi/{vid}/hqdefault.jpg'
                
                
                with ui.card().classes('w-80 mx-auto cursor-pointer') as card:

                
                    ui.image(thumbnail).classes('w-full h-48 bg-gray-200')
                
                    with ui.card_section():
                        ui.label(title).classes('text-h6 font-medium')
                    
                card.on('click',partial(update_page.show_video_player,vid))

                    

        app.storage.general['videos_shown']=end

        if end>=len(video_ids):
            load_more_button.visible =False

            with context_container:
                ui.label('no more videos!')
    ui.timer(0.1, load_more, once=True) #display first 21 videos 
    load_more_button.on_click(load_more)




@ui.page('/')
def website():
    '''the function that gather all other funtion to run the website'''
    

    app.storage.general['videos_shown'] = 0
    video_per_load=21

    video_play_container=ui.column().classes('w-full')
    feed_container=ui.column().classes('w-full')

    
    page_update=UpdateWebsitePage(video_play_container,feed_container)

    
    
    with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('Vrdndi')

            if ENABLE_EXPERIMENTAL_FEATURES:
                feed_button=ui.button('New Feed Availabl!', on_click=refresh_website_for_new_feed).classes('ml-auto')\
                .bind_visibility_from(new_feed_state,'feed_state')
    

    with feed_container:
        render_feed(update_page=page_update,video_per_load=video_per_load)


    

if __name__ in {"__main__", "__mp_main__"}:

    ui.run(
        host=WEBSITE_HOST,
        port=8080,
        reload=False,
        storage_secret='idkwhatever',
        reconnect_timeout=10
    )



