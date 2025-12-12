'''
Set up the data,artifacts,secrets folder and define some other path
'''

from pathlib import Path


current_file_path=Path(__file__)
PROJECT_ROOT=Path(__file__).parent.parent

ARTIFACTS_PATH=PROJECT_ROOT/'artifacts'

SECRET_PATH=PROJECT_ROOT/'secrets'


DATA_PATH=PROJECT_ROOT/'data'

RAW_DATA_PATH=DATA_PATH/'raw'

PROCESSED_DATA_PATH=DATA_PATH/'processed'

TRAIN_DATA_PATH=PROCESSED_DATA_PATH/'train'
INFERENCE_DATA_PATH=PROCESSED_DATA_PATH/'inference'

DATABASE_PATH=DATA_PATH/'database'



dirs_to_create= [
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    INFERENCE_DATA_PATH,
    DATABASE_PATH,
    SECRET_PATH,
    ARTIFACTS_PATH,
]

for dir in dirs_to_create:
    dir.mkdir(parents=True,exist_ok=True)




TEST_PATH=PROJECT_ROOT/'test'

FIXTURE_PATH=TEST_PATH/'fixture'


SRC=PROJECT_ROOT/'src'

ASSETS_PATH=SRC/'assets'


LEGACY_PATH=PROJECT_ROOT/'legacy'