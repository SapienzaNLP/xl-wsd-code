conda create --name xl-wsd-code python=3.7
conda activate xl-wsd-code
pip install -r requirements.txt
conda install pytorch==1.5.0 torchtext cudatoolkit=10.1 -c pytorch
