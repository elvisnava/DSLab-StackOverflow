# this assumes you have a virutal environment or conda or something activated

# first install requirements. including an old version of tensorflow
pip install -r requirements.txt
note I changed tensorflow from 1.2.0 to 1.13.0 since the former is not on pip

go to https://github.com/GPflow/GPflow/tree/0.4.0 (make sure the tag 0.4.0 is activated)
download the repo as zip somewhere on your computer (not into our repo)

open a terminal in the root of GPflow
make sure the correct virtualenv is activated
run 'pip install .'