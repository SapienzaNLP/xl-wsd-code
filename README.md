# XL-WSD
Code for the paper [XL-WSD: An Extra-Large and Cross-Lingual Evaluation Frameworkfor Word Sense Disambiguation](). Please visit https://sapienzanlp.github.io/xl-wsd/ for more info and to download the data.

# Install
First setup the python environment.
```bash
conda create --name xl-wsd-code python=3.7
conda activate xl-wsd-code
pip install -r requirements.txt
conda install pytorch==1.5.0 torchtext==0.6.0 cudatoolkit=10.1 -c pytorch
```
Then, download and install WordNet.
```bash
cd /tmp
wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz
tar xvzf WordNet-3.0.tar.gz
sudo mv WordNet-3.0 /opt/
rm WordNet-3.0.tar.gz
```
In case you do not want to move WordNet to the `/opt/` directory, then set the `WORDNET_PATH` variable in `src/datasets/__init__.py` to the full path of your WordNet-3.0 directory.

# Train

# Evaluate
