

## Add API Key

**Step 1:**
Go to google cloud to make a project and enable the ``youtube api v3`` and get your client file. Please place the client file in the ``secrets/`` folder.

**Step 2**: 
And copy file name of your client secret to ``CLIENT_SECRET_FILE`` in ``src/config.py``


## Save Your Data To Database

Open ``scripts/data_saving.py`` and run the file. It would save your like,dislike palylist video data to database and fech the youtube video data from your subscription. (It won't save all videos of your subscriber for now, but it would still be a lot )


And you could check the database to see if the data is enough for training, say total datapoints are larger than 500 (The least playlist data should be larger than 100 to get a better starting point)


## Streamlit Data Labelling

If you suspect the amount of data isn't enough, you could run the ``datalabel_streamlit.py`` to collect more data from your history

**At first**, go to Google Takeout and export your youtube history data. It should be named as ``watch-history.json``, if it's not, please rename to that. And move it inside ``data/raw`` folder

**Second**, go back to ``scripts/data_saving.py``, remove the function you just called, call ``get_and_clean_his_video_data()`` function (It should be already imported). The function would clean your history data and save it to database.

**Now**, you could run ``datalabel_streamlit.py`` in your **Terminal**.

For example:

```bash
cd vrdndi/scripts
streamlit run datalabel_streamlit.py
```
**Screenshot**:
![[streamlit screenshot]](images/streamlit_screenshot.png)

**Usage Explain**
* **Skip**: Skip current video to next video
* **Save**: Save current progress. It woud save current video index and labelled data to database
* **Previous**: Move back to previous video.
* **Undo**: Remove last video you labelled.
* **Load data**: Load your progress from database.
* **interest**: Label current video as interesting.
* **uninterest**: Label current video as uninteresting.


## Data Preprocess
For clarity and simplicity , I would call the data from your like,dislike playlist and streamlit as *interest data*; the feedback data from NiceGUI website as *productive data* in later part of this document.

**Save interest data**

Open ``data_saving.py``, change the function to ``like_dislike_streamlit_data_preprocess()`` and run the file. It would save intereset data to database.

**Save training data**

>If you followed the above step, and this is first time to use this project, **skip** this part. You can't save it now, because you haven't have productive data, yet. 

Open ``data_saving.py``, change the function to ``interest_productive_data_preprocess()`` and run the file. It would use the videoId in your feeback to fetch data from Youtube API and save it to database.


## Configuration

## Training

## Inference