


# Vrdndi

Vrdndi (Verdandi) is a full stack recommendation system that process your media data (Youtube currently) to provide personal feed base on what you did previously in your computer (i.e dynamicly change the feed base on time and previous app history)

The original goal of this project is not to increase your watching time in your feed like other recommendation system. It's the opposite, minimize your watch time, increase your productivity but keep your interest still (Currently it won't work in such way, see Limitation section)



![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![SQLite3](https://img.shields.io/badge/SQLite-07405E?style=flat&logo=sqlite&logoColor=white)
![ActivityWatcher](https://img.shields.io/badge/ActivityWatcher-1D1D1D?style=flat&logo=activity&logoColor=white)
![NiceGUI](https://img.shields.io/badge/NiceGUI-5898d4?style=flat&logoColor=white)



## Installation

Step 1: Clone the repository

```bash
npm install vrdndi
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

For the ones who may wonder, the reason I used AdaLN instead of normal LN is because the duration is a numerical value, it can't go through the BGE-M3's embedding layer, so either I need to put it as a separate token or put it as a condition to diffuse the AW data. I chose the latter.  And use SE block to "gate" each token to pick the important one. And I replace the common GLU with SiLY to be consistant with SWiGLU, since it's almost interchangable.


Residual Block: 

![[Residual block ]](docs/images/Residual_block_structure.svg)

And there's two head as the output layer of the model: interest head and productive head. The interest head would use as a trainsition before you have enough productive data(i.e. The data you labelled in the website) and the productive head to predict a rate base on previous app history, time for each media data. And the reason why I used SWiGLU instead of normal ReLU is that previously the interest head can't quite converge (at least the bouncing range is larger than now), and since the sequence compressor for interest head is kinda partial functional (It won't receive a app sequence to predict interest, so the output token would just represent the duration that diffused in it). So probably adding a strong activation function in output layer would be a good idea, and I also switch the productive head to SWiGLU at that time as convient, but seemingly it cause the overfitting problem that is faced on currently.  


Output Layer:
![[Output layer]](docs/images/output_layer.svg)

## Model Performance

I think the performance is not bad, one of the all 5 folds could hit 0.95 f1, which is suspiciously high. But since it only had one fold hitted that, so for now I'm fine with that. And the model structure is decent enough as the test may prove.

Productive head's mean F1 performance with Standard Deviation:

![[Prodcuctive head's performance]](docs/images/productive_val_f1_with_std.svg)

More performance detail:

![[detail performance chart]](docs/images/overall_model_performance_chart.png)

## Hardware requirement

I don't know the exact performance requirement of this project, but I could give your some references:

- RTX 3060 laptop with 16 GB RAM would be perfectly fine with this project.
- M1 Macbook Air with 16 GB RAM could run the inference, but may not be able to train the model


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

- Refine the feed displaying method (i.e. not just display the highest rated items)

- Rewrite the website 

## Optimizations

- Improve code clarity and readability

- Fix the productive inference test function


## Limitation

As you may notice, current project doesn't do anything about the original goal. It's more like keeping or organizing your feed as you want, even if it's not a productive way. In the future version, we may reach that goal.




## File Strcuture

```
.
├── artifacts                                The location of model file 
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







