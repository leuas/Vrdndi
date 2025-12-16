## Contents
1. [Prerequisites](#prerequisites)

2. [Setup Activity Watcher (AW)](#setup-activity-watcher-aw)
3. [Add API Key](#add-api-key)
4. [Save Your Data To Database](#save-your-data-to-database)
5. [Streamlit Data Labelling](#streamlit-data-labelling)
6. [Data Preprocess](#data-preprocess)
7. [Configuration](#configuration)
    - [Base Productive Model](#base-productive-model)
    - [Hybrid Productive Model](#hybrid-productive-model)
    - [Usage Example](#usage-example)
8. [Training](#training)
9. [Inference](#inference)
10. [NiceGUI Website](#nicegui-website)
11. [Scheduler](#scheduler)
12. [Remote Connect (Optional)](#remote-connect-optional)
13. [Data Transfer (Optional)](#data-transfer-optional)



## Prerequisites

* **Installed** project via the guide in ``README.md``

* **Sign in** to Wandb. You could wait to do so until you run the training pipelines. Currently we use Wandb to log your training metrics and since we just save the LoRA layers (roughly 20 MB), so the 5 GB free storage is enough. 
>I am considering switching to ``mlflow`` later, but for now, sorry for introducing a third-party service.



## Setup Activity Watcher (AW)

**Step 1**: Install [ Activity Watcher](https://github.com/ActivityWatch/activitywatch) 

**Step 2**: Just wander around or watch some videos on your computer while the Activity Watcher is running.

**Step 3**: Go to its dashboard. Click ``Settings``, scroll down, find [Category Builder ](http://localhost:5600/#/settings/category-builder) (or click this) and set up some categories for what you did.

>**Why:** The function (``get_aw_raw_data``) that fetches data from AW  uses the ``get_classes`` function from ``aw_client.classes`` and sometimes if you haven't set up the categories, the fallback of ``get_classes`` to default class might not work correctly sometimes. (It's possible because I used the function incorrectly)
>
>So it's better to just set up the categories first to avoid that error. I've added this to the backlog, and might investigate and add an error handling in a future version.

**Step4**: Change the ``HOSTNAME`` in ``src/config.py`` to your actual hostname. You can find that in the AW dashboard.

Example:
```python
HOSTNAME='randompersonMacBook.local'
```



## Add API Key

**Step 1:**
Go to Google Cloud, create a project and enable the ``YouTube API v3`` and download your client secret file(credentials file). Please place the client file in the ``secrets/`` folder.

**Step 2**: 
Copy file name of your client secret into ``CLIENT_SECRET_FILE`` in ``src/config.py``


## Save Your Data To Database

Open ``scripts/data_saving.py`` and run the file. It will save your like,dislike playlist video data to database and fetch the YouTube video data from your subscriptions. (It won't save *every* videos of your subscriber for now, but it will likely still be a lot )


And you can check the database to see if the data is enough for training. For example, total datapoints are larger than 500 (The smallest playlist data should have at least 100 items to get a better starting point)


## Streamlit Data Labelling

If you suspect the amount of data isn't enough, you can run the ``datalabel_streamlit.py`` to collect more data from your YouTube history

**First**, go to Google Takeout and export your YouTube history data. The file should be named `watch-history.json` (Please rename it if it differs) And move it inside ``data/raw`` folder.

**Second**, go back to ``scripts/data_saving.py``. Remove the function you just called, and call ``get_and_clean_his_video_data()`` function (It should already be imported). The function cleans your history data and save it to database. (If it successed, you could delete your history file freely)

**Now**, you can run ``datalabel_streamlit.py`` in your **Terminal**.

For example:

```bash
cd vrdndi/scripts
streamlit run datalabel_streamlit.py
```
**Screenshot**:
![[streamlit screenshot]](images/streamlit_screenshot.png)

**Usage Explain**
* **Skip**: Skip current video to next video
* **Save**: Save current progress. It writes current video index and labelled data to database
* **Previous**: Move back to previous video.
* **Undo**: Remove last video you labelled.
* **Load data**: Load your progress from database.
* **interest**: Label current video as interesting.
* **uninterest**: Label current video as not interesting.


## Data Preprocess
For clarity and simplicity , I would refer to the data from your like,dislike playlist and streamlit as *interest data*; the feedback data from NiceGUI website as *productive data* in later part of this document.

**Save interest data**

Open ``data_saving.py``, change the function to ``like_dislike_streamlit_data_preprocess()`` and run the file. It would combine all these three data and save it into ``interest_data`` table in database.

>**Note**: As the name suggests, you may need to label some data in streamlit before running this function, or currently you could delete `streamlit_data=db.get_data('streamlit_data')`line in that function. 
>
> Added to backlog to check if table if exist and hanle that in that function. 

**Save training data**

>**Note**: If you followed the above step, and this is first time to use this project, **skip this part**. You can't save it now, because you haven't labelled productive data, yet. 

Open ``data_saving.py``, change the function to ``interest_productive_data_preprocess()`` and run the file. It would use the `videoId` in your feedback to fetch data from YouTube API and save it to database.


## Configuration

For reference, see [Configuration](CONFIGURATION.md).

### Usage Example

Currently all type of hyperparameters are mixed in two classes: ``ProductiveModelConfig`` and ``HybirdProductiveModelConfig``. I may organize them in future.


```python
from src.pipelines.productive import HybridProductiveModelTraining
from src.config import HybridProductiveModelConfig

config=HybridProductiveModelConfig()

config.train_num_workers=4
config.eval_test_num_workers=4
config.accumulation_steps=4

config.interest_loss_weight=0.33

config.sampler_interest_ratio=3/4
config.productive_output_layer_dropout=0.5

test=HybridProductiveModelTraining(config=config)

```


## Training


After you have the training data, just adjust the config parameter and run it.

**Example (Follow with previous example)**:

```python
test.start_train(model_name='example.pth')
```

If you want to test model performance, you could also use K-Fold training, but currently it won't save the model.

```python
test.kfold_start()
```

**Note**

The model saving part would override model file that saved in previous epoch, if its EMA f1 is lower than the model in current epoch. And if you forgot to change the model saving name, it's fine, there's a check function before running the actual training process.

Also If you saw the warning `` Argument path is not specified. Defaulting to TRAIN_DATA_PATH ``, yes, it's ecptected output
## Inference


``prepare_predicting_data()`` would fetch the video data in a time range from now, say last 7 days in video data you saved from database and encode the app history sequence and save it to ``data/processed/inference``.


``predict()`` would predict the data it receive and save it to database.

**Example**
```python
from src.inference.productive import HybirdProductiveModelPredicting
from src.config import HybirdProductiveModelConfig

config=HybirdProductiveModelConfig()
config.eval_test_num_workers=8

model=HybirdProductiveModelPredicting('example.pth',config=config)

inference_data=model.prepare_predicting_data()

model.predict(inference_data=inference_data) 
```

>Note: I haven't add a function to automatically clean the encoded tensor file (but added to backlog), so you may delete these manually. 


## NiceGUI Website

You could run the website in the **terminal**.

Example:
```bash
cd vrdndi/src/web
python website_frontend.py
```

And the feedback you gave in the website would save to ``feedback`` table in database.

>**Note**: You should run the inference to get the feed in the database before running the website.


## Scheduler

You could run the ``sheduler.py`` in the ``scripts/`` to update your feed regularly.

The updated feed would save to database, reload the website you could see the new feed.




## About Baseline

Just don't touch it for now. I haven't connect it to database yet. So you could say in current version, it's unusable.



## Remote Connect (Optional)

If you want to train your model in the computer you are not usually used. You could use Tailscale

Example:
Say, I usually use Macbook, but I want to train it in a windows computer, you could use Tailscale to give your two computer a specific IP address to connect each other.

Set ``HOST`` to the ip of the computer you usually use ( the one you run Activity Watcher )

```python
HOST='100.100.666.42'
```

## Data Transfer (Optional)

If you want to move your training data to another computer, I recommend you to use DVC. BUt copy-paste in Teamviewer or whatever other method is totally fine.
