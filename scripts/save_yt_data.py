'''save your youtube data to database '''


from src.utils.data_etl import get_and_save_liked_disliked_data_for_database,get_and_save_yt_video_for_database,get_and_save_his_data_for_database



if __name__ == '__main__':
    get_and_save_liked_disliked_data_for_database()
    get_and_save_yt_video_for_database()
    