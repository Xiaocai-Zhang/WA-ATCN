# WA-ATCN for Traffic Time-Seris Forecasting
## Setup
Code was developed and tested on Ubuntu 18.04 with Python 3.6 and TensorFlow 2.5.0. You can setup a virtual environment by running the code like this:
```
python3 -m venv env
source env/bin/activate
cd env
pip3 install -r requirements.txt
```
## Running Models
You can run the following commands to replicate the results.
```
python3 script/test_m50.py
python3 script/test_i280.py
python3 script/test_nyc.py
```
The results are saved under the "./para/" folder.
## Training Models
You can run the following commands to train models.
```
python3 script/train_m50.py
python3 script/train_i280.py
python3 script/train_nyc.py
```

