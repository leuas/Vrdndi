

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

**At first**. Go to Google Takeout and export your youtube history data. It should be named as ``watch-history.json``, if it's not, please rename to that.

**Second**, go back to ``scripts/data_saving.py``, remove the function you just called, call ``get_and_clean_his_video_data()`` function (It should be already imported). The function would clean your history data and save it to database.

**Now**, you could run ``datalabel_streamlit.py`` in your **Terminal**.

For example:

```bash
cd vrdndi/scripts
python datalabel_streamlit.py
```


**Step 2**: 
Run the train.py to train the model, you could adjust the name of the model, etc. 

```bash
python train.py 
```

**Step 3**:
Run the scheduler to setup the website and update the feed in the range of time

```bash
python scheduler.py
```


**Optional**: Model would use your like and dislike video list as its training data to train the interest head, and probably it won't be enough of data. And you may use the streamlit script in the scripts folder to label the data by yourself. The script would use your youtube history data as its source and you could review your history video and label if it's interesting. Personally, the lableing process is quite fun, so probably you would like it too.


## Configuration

## Training

## Inference