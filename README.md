


# Vrdndi

Vrdndi (Verdandi) is a full stack recommendation system that process your media data (Youtube currently) to provide personal feed base on what you did previously in your computer (i.e dynamicly change the feed base on time and previous app history)


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![SQLite3](https://img.shields.io/badge/SQLite-07405E?style=flat&logo=sqlite&logoColor=white)
![ActivityWatcher](https://img.shields.io/badge/ActivityWatcher-1D1D1D?style=flat&logo=activity&logoColor=white)
![NiceGUI](https://img.shields.io/badge/NiceGUI-5898d4?style=flat&logoColor=white)



## Installation

Step 1: Clone the repository

```bash
  npm install my-proje
  cd my-project
```

Step 2: Install the package. You may need to adjust the hyperparameter of the model, so install in editable mode

```bash
  pip install -e . 

```


## Model Structure

Currently Vrdndi uses fine-tuned BGE-m3 to predict the media feed. 

There's two type of input, the media data data(i.e. title or description of a video) and the app sequence data from Activity Watcher. Video data would directly go into the model embedding layer, but app sequence won't, I made a custom residual block sequencial to preocess it and encode the sequence into one token. 

Main Structure: 
![[Main structure]](docs/images/Model_main_structure.svg)

For the ones who may wonder, the reason I used AdaLN instead of normal LN is because the duration is a numerical value, it can't go through the BGE-M3's embedding layer, so either I need to put it as a separate token or put it as a condition to diffuse the AW data. I chose the latter.  And use SE block to "gate" each token to pick the important one.


Residual Block: 

![[Residual block ]](docs/images/Residual_block_structure.svg)


## Model Performance

So the picture below is the model performance in k-fold (n_splits equal to 5) under 200-300 rougly productive datapoints (plus 1000 interest data). I think the performance is not bad, one of the all 5 folds could hit 0.95 f1, which is pretty high. 

![[Prodcuctive head's performance]](docs/images/productive_val_f1_with_std.svg)

More performance detail:

![[detail performance chart]](docs/images/overall_model_performance_chart.png)

## Usage/Examples


Step 1: Go to google api to make a project and get your client file. Please place the client file in the secret folder.


Step 2: 
Run the train.py to train the model, you could adjust the name of the model, etc. 

```bash
  python train.py 
```

Step 3:
Run the scheduler to setup the website and update the feed in the range of time

```bash
  python scheduler.py
```


Optional: Model would use your like and dislike video list as its training data to train the interest head, and probably it won't be enough of data. And you may use the streamlit script in the scripts folder to label the data by yourself. The script would use your youtube history data as its source and you could review your history video and label if it's interesting. Personally, the lableing process is quite fun, so probably you would like it too.

## Website
The website is functional as watching video and scrolling the feed and giving feedback.

As you may know, I used NiceGUI to create this website. And even tho it's funtional, probably I would rewrite the website later on :) 

## Roadmap


- Add youtube transcript as one of the model feature

- Add video upload time as the condition of AdaLN

- Add more media data (e.g. Email, RSS feed,etc) 

- switch to mlflow from wandb

- Rewrite the website 

## Optimizations

- Improve code clarity

- Add more error handling


## Project Status

Currently, the model performance would be consistently higher than 0.8, but I'm uncertain 



## File Strcuture

```
.
├── artifacts
├── data
│   ├── database
│   ├── processed
│   │   ├── inference
│   │   └── train
│   └── raw
├── docs
├── pyproject.toml
├── pytest.ini
├── scripts
│   ├── datalabel_streamlit.py
│   ├── debug.py
│   ├── run.py
│   ├── scheduler.py
│   ├── train.py
│   └── visualise_model.py
├── secrets
├── src
│   ├── __init__.py
│   ├── assets
│   │   └── hit_stopwords.txt
│   ├── config.py
│   ├── db
│   │   ├── __init__.py
│   │   └── database.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   └── productive.py
│   ├── model_dataset
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── productive.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── activity_watcher_encoder.py
│   │   ├── components.py
│   │   └── productive.py
│   ├── pipelines
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   └── productive.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── data_etl.py
│   │   └── ops.py
│   └── web
│       ├── __init__.py
│       └── website_frontend.py
└── test 


```






