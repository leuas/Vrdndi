


# Vrdndi

Vrdndi (Verdandi) is a full-stack recommendation system that process your media data (Youtube currently) to provide personal feed base on what you did previously in your computer (i.e dynamicly change the feed base on time and previous app history)

The primary goal of this project is not to increase your watching time in your feed like other recommendation system. It's the opposite, minimize your watch time, increase your productivity but keep your interest still 
>Currently it won't work in such way, see [ Limitation section](#limitation)



![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=plastic&logo=Streamlit&logoColor=white)
![SQLite3](https://img.shields.io/badge/SQLite-07405E?style=plastic&logo=sqlite&logoColor=white)
![ActivityWatcher](https://img.shields.io/badge/ActivityWatcher-%23ffff?style=plastic&logo=activity)
![NiceGUI](https://img.shields.io/badge/NiceGUI-5898d4?style=plastic&logoColor=white)



## Installation

**Step 1**: Clone the repository

```bash
git clone https://github.com/leuas/Vrdndi.git
cd vrdndi
```

**Step 2**: Install pytorch for your GPU

Go to [Pytorch Get Started](https://pytorch.org/get-started/locally/) and pick the version that suit your computer.

For example:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Step 3**: Install the package. You may need to adjust the hyperparameter of the model, so install in editable mode.

```bash
pip install -e . 
```

## System Architecture

Current system Architecture as the picture shown would fetch the data from Youtube API and your ActivityWatcher to predict output feed and save it into database. Website would fetch the feed from database to render videos.

**End-to-End Pipelines**:

![[system_architecture]](docs/images/System_Architecture_big.svg)



## Model Structure

Currently Vrdndi uses fine-tuned BGE-m3 to predict the media feed. 

There's two type of input:

1, **Media Data**: Textual data (Title/Description). Directly processed by the BGE-M3 embedding layer.

2, **App Sequence Data**: Activity history from ActivityWatcher. Processed by a custom sequential residual block to encode the sequence into a single token.
> Sequence data would be pre-computed before running the main model to save memory usage. (Text part would be encoded by Sentence Transformer)


**Main Structure**: 

![[Main structure]](docs/images/Model_main_structure.svg)

>**Why AdaLN:** Duration is a numerical value, it can't go through the BGE-M3's embedding layer, so either I need to put it as a separate token or put it as a condition to diffuse the AW data. In former, seemingly it would cause distribution mismatch(?), so I use the latter.


And I used pre-activation structure instead of post-activiation one for the Residual block. (As we know, to let the gradient to flow back more easily) I also added SE block to "gate" each token to pick the important one and replaced the common GLU with SiLU to be consistant with SWiGLU, since it's almost interchangable.

**Residual Block**: 

![[Residual block ]](docs/images/Residual_block_structure.svg)

And there's two head as the output layer of the model: interest head and productive head. The interest head would use as a trainsition before you have enough productive data(i.e. The data you labelled in the website) and the productive head to predict a rate base on previous app history, time for each media data. 


**Output Layer**:

![[Output layer]](docs/images/output_layer_big.svg)

>**Why SWiGLU:** Previously the interest head can't quite converge (at least the bouncing range is larger than now), and since the sequence compressor for interest head is kinda partial functional (It won't receive a app sequence to predict interest, so the output token would just represent the duration that diffused in it). So probably adding a strong activation function in output layer would be a good idea, and I also switch the productive head to SWiGLU at that time as convenient, but seemingly it cause the overfitting problem that is faced on currently.  

## Model Performance

I think the performance is not bad, one of the all 5 folds could hit 0.95 f1, which is suspiciously high. But since it only had one fold hitted that and my dataset is quite small (200-300 for productive,1000 roughly for interest), so for now I'm fine with that. And the model structure is decent enough as the test may proved.

Productive head's mean F1 performance with Standard Deviation:

![[Prodcuctive head's performance]](docs/images/productive_val_f1_with_std.svg)

More performance detail:

![[detail performance chart]](docs/images/overall_model_performance_chart.png)

## Hardware requirement

Some references:

- RTX 3060 laptop with 16 GB RAM would be perfectly fine with this project.
- M1 Macbook Air with 16 GB RAM could run the inference, but may not be able to train the model


## Usage/Examples

Quick start:

Show the basic model inference. For detail adjustment of demo, please see the docstring in ``demo.py``

```bash
cd Vrdndi/scripts

python demo.py
```

For detail or general usage, please see [Usage Guide](docs/USAGE.md)

## Privacy Notes
**Data Privacy**

All data that's used in this project is processed locally. It's in the ``data/`` folder. You have full control over it.

**Internet Requirement**
* **Pipelines**: If you download the base BGE-M3 model, Sentence Transformer and its tokenizers, you could run it without Internet.

* **Website**: The local website need to access to Internet to render Youtube video. 


## Website
The NiceGUI website is functional as watching video and scrolling the feed and giving feedback.

Main page would render 21 videos at once, you could press the ``LOAD MORE`` button to get more videos



**Main Page**:
![[main page]](docs/images/Main_page.png)

**Video Page**:
![[video page]](docs/images/Video-play_page.png)

>For streamlit data labeling website, please see [Usage Guide](docs/USAGE.md)

## Backlog

- Add youtube transcript as one of the model feature

- Add video upload time as the condition of AdaLN

- Add more media data (e.g. Email, RSS feed,etc) 

- Refine the feed displaying method (i.e. not just display the highest rated items)

- Improve appearance of the website

- Display original video title instead of lowercase one

- Write a automatic function to clean offline tensor files

- Fix the productive inference test function

- More but haven't listed 



## Limitation
Technically speaking, it *can* aim for productivity if your have enough data and its quality is really good. But it's really hard to reach that since current state of the project doesn't do anything to explicitly form the feed in a way to achieve the primary goal.

It's more like keeping or organizing your feed as you want, even if it's not in a productive way. In the future version, we may reach that goal. (Say add RL?)

Also I'm skpetical about whether the current pipelines and model architecture would really previde a good quality of feed, while at some point the performance at small dataset is not bad. Probably we may know that when more data are collected.

## File Strcuture

```
.
├── artifacts/                              # Model artifacts                           
├── data                                     
│   ├── database                            # SQLite database files
│   ├── processed           
│   │   ├── inference                       # Offline encoded tensors for inference
│   │   └── train                           # Offline encoded tensors for training
│   └── raw                                 # Raw data(e.g. watch-history.json)
├── docs/                                   # Documentation and images
├── pyproject.toml
├── pytest.ini
├── scripts/                                # Demo, training and sheduler scripts
├── secrets/                                # API token and client sercrets (put yours there)
├── src
│   ├── assets/                             # Stopwords
│   ├── config.py                           # Global configuration parameters
│   ├── path.py                             # Path constants
│   ├── db/                                 # Database class
│   ├── inference                           # Inference
│   │   ├── baseline.py
│   │   └── productive.py                   # Main model inference class
│   ├── model_dataset                       # Dataset
│   │   ├── loader.py                       # Dataloader class
│   │   └── productive.py                   # Dataset class of main model
│   ├── model
│   │   ├── activity_watcher_encoder.py     # The model that encode text of the app sequence 
│   │   ├── components.py                   # Defines some model layers (e.g. AdaLN)
│   │   └── productive.py                   # Main model class
│   ├── pipelines                           # Training pipelines
│   │   ├── baseline.py
│   │   └── productive.py                   # Pipelines of main model
│   ├── utils
│   │   ├── data_etl.py                     # Processes data from api
│   │   └── ops.py                          # Data operation for model training or inference
│   └── web
│       └── website_frontend.py             # NiceGUI website frontend
└── test 

```


## Notes

Since this is my very first project in code (except code practise), and I'm still new to programming and ML. I may miss something completely.

If you find any bug/issue or problem about the architecture (Say, the architecture cause some converage problems) or anything else. Please feel free to open an issue in Github! (or even submit a PR!)





