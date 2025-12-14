

## Common Usage

**Step 1**: Go to google api to make a project and get your client file. Please place the client file in the secret folder.


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