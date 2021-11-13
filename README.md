# WA-ATCN for Traffic Time-Seris Forecasting
## Setup
Code was developed and tested on Ubuntu 18.04 with Python 3.6 nnd TensorFlow 2.5.0
```
python3 -m venv env
source env/bin/activate
cd env
pip3 install -r requirements.txt
```
## Test
```
python3 script/test_m50.py
python3 script/test_i280.py
python3 script/test_nyc.py
```
## Train Models
```
python3 script/train_m50.py
python3 script/train_i280.py
python3 script/train_nyc.py
```
