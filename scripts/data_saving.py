'''save your data to database '''

import logging
from src.utils.data_etl import get_and_save_liked_disliked_data_for_database,get_and_save_yt_video_for_database,get_and_save_his_data_for_database
from src.utils.ops import like_dislike_streamlit_data_preprocess,interest_productive_data_preprocess


if __name__ == '__main__':


    get_and_save_liked_disliked_data_for_database()
    get_and_save_yt_video_for_database()
    