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

* **Sign in** to Wandb. You could suspend to do so before you run the training pipelines. Currently we use Wandb to log your training metrics and since we just save the LoRA layers (20 MB roughly) of the models, so the 5 GB free storage is enough. 
>I am considering switch to ``mlflow`` later, but for now, sorry for introducing third party serivce.



## Setup Activity Watcher (AW)

**Step 1**: Install [ Activity Watcher](https://github.com/ActivityWatch/activitywatch) 

**Step 2**: Just wonder around or watch some videos in your computer when the Activity Watcher is running.

**Step 3**: Go to its dashboard. Click ``Settings``, scroll down, find [Category Builder ](http://localhost:5600/#/settings/category-builder) (or click this) Setup some category for what you did.

>**Why:** The function that fetch data from AW (``get_aw_raw_data``) use the ``get_classes`` function from ``aw_client.classes`` and sometimes if you didn't setup the category, the fallback of ``get_classes`` to use default class won't work correctly somehow. (It's possible that it's because I used that function in a wrong way)
>
>So it's better that you just setup the category first to avoid that error. 
>
>I already added that to backlog, may investigate and add an error handling to that in later version.

**Step4**: Change the ``HOSTNAME`` in ``src/config.py`` to your actual hostname. You could saw that in the AW dashboard.

Example:
```python
HOSTNAME='randompersonMacBook.local'
```



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

Open ``data_saving.py``, change the function to ``like_dislike_streamlit_data_preprocess()`` and run the file. It would combine all these three data and save it into ``interest_data`` table in database.

>**Note**: As the name goes, you may need to label some data in streamlit before running this function, otherwise

**Save training data**

>**Note**: If you followed the above step, and this is first time to use this project, **skip** this part. You can't save it now, because you haven't have productive data, yet. 

Open ``data_saving.py``, change the function to ``interest_productive_data_preprocess()`` and run the file. It would use the videoId in your feeback to fetch data from Youtube API and save it to database.


## Configuration

### Base Productive Model

The Basic configuration

| Parameter   | Type    | Default     | Description                                                 |
| --------------- | ----------- | --------------- | --------------------------------------------------------------- |
| `model_name`    | `str`       | `'BAAI/bge-m3'` | The base pre-trained model. Usually you won't change it |
| `seed`          | `int`       | `42`            | Random seed  |
| `compile_model` | `bool`      | `True`          | Whether to use `torch.compile` for faster inference/training.   |
| `batch_size`    | `int`       | `4`             | Number of samples processed per training step.                  |
| `total_epoch`   | `int`       | `10`            | Total number of training epochs.                                |
| `g`             | `Generator` | `None`          | Placeholder, it would be updated in the pipelines        |
|||||

The other configuration of ``ProductiveModelTraining``
|||||                  
| --------------------------------- | -------- | ----------- | ----------------------------------------------------------
| `lr`           | `float`  | `5e-5`      | Learning rate for the optimizer.                                    |
| `weight_decay` | `float`  | `1e-3`      | L2 penalty applied to weights to prevent overfitting.               |
| `ignore_index` | `int`    | `-100`      | Label index to ignore when calculating loss and weight. |
| `wandb_config` | `dict`   | _{lr}_      | Dictionary configuration for Weights & Biases logging.              |
|||||

The Configuration of ``ProductiveModel``

|||||                  
| --------------------------------- | -------- | ----------- | ---------------------------------------------------------- |
| `productive_out_feature`          | `int`    | `2`         | Output dimension for the productive classification head. |
| `interest_out_feature`            | `int`    | `2`         | Output dimension for the interest classification head.   |
| `productive_output_layer_dropout` | `float`  | `0.1`       | Dropout rate applied to the productive head.               |
| `interest_output_layer_dropout`   | `float`  | `0.1`       | Dropout rate applied to the interest head.                 |
|||||

THe LoRA part of ``ProductiveModel``

|||||
|---|---|---|---|
|`use_lora`|`bool`|`True`|Enable parameter-efficient fine-tuning using LoRA.|
|`lora_rank`|`int`|`8`|The dimension (r) of the low-rank matrices.|
|`lora_alpha`|`int`|`16`|Scaling factor for LoRA weights.|
|`lora_target_modules`|`str`|`'all-linear'`|Which modules to apply LoRA to (e.g., `q_proj`, `v_proj`).|
|||||

For Loss & EMA:

|||||
|---|---|---|---|
|`productive_loss_weight`|`float`|`1`|Weight multiplier for the productive head loss.|
|`interest_loss_weight`|`float`|`1`|Weight multiplier for the interest head loss.|
|`ema_alpha`|`float`|`0.6`|Smoothing factor for Exponential Moving Average.|
|`ema_productive_weight`|`float`|`0.65`|Specific weight factor applied during EMA calculations.|
|||||

### Hybrid Productive Model

>**Note**: It's the child class of ``ProductiveModel``, so it inherit all the configuration (above) from base model.


| Parameter     | Type | Default | Description                                                     |
| ----------------- | -------- | ----------- | ------------------------------------------------------------------- |
| `num_in_feature`  | `int`    | `3`         | Number of additional input features (Duration, Time Sin, Time Cos). |
| `num_out_feature` | `int`    | `384`       | Number of dimension in projected output tensor. In Default, it's same with encoded textual tensor by Sentence Transformer                        |
| `cond_dim`        | `int`    | `1`         | Dimension size for the conditional input (Duration for now).                |
| `max_length`      | `int`    | `8094`      | Maximum sequence length (tokens) allowed for input. I set it to the maximum of BGE-M3. In most case, it won't hit that                |
|                   |          |             |                                                                     |

For Training & Sampling
|                   |          |             |                                                                     |
|---|---|---|---|
|`accumulation_steps`|`int`|`4`|Number of steps to accumulate gradients before updating weights.|
|`sampler_interest_ratio`|`float`|`0.5`|Ratio of "Interest" samples in a training batch.|
|`sampler_productive_ratio`|`float`|`0.5`|Ratio of "Productive" samples (Calculated as `1 - interest_ratio`).|
|`interest_label_smooth`|`float`|`0`|Label smoothing factor for the Interest loss function.|
|`productive_label_smooth`|`float`|`0`|Label smoothing factor for the Productive loss function.|


### Usage Example

Currently all type of hyperparameters is mixing in two classs: ``ProductiveModelConfig`` and ``HybirdProductiveModelConfig``. May organize them in future.


```python
from src.pipelines.productive import HybirdProductiveModelTraining
from src.config import HybirdProductiveModelConfig

config=HybirdProductiveModelConfig()

config.train_num_workers=4
config.eval_test_num_workers=4
config.accumulation_steps=4

config.interest_loss_weight=0.33

config.sampler_interest_ratio=3/4
config.productive_output_layer_dropout=0.5

test=HybirdProductiveModelTraining(config=config)

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

The model saving part would override model file that saved in previous epoch, if its EMA f1 is lower than the model in current epoch.

If you forgot to change the model saving name, it's fine, there's a check function before running the actual training process.

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
