# XL-WSD
Code for the paper [XL-WSD: An Extra-Large and Cross-Lingual Evaluation Frameworkfor Word Sense Disambiguation](). Please visit https://sapienzanlp.github.io/xl-wsd/ for more info and to download the data.

# Install
First setup the python environment. Be sure that [anaconda](https://docs.anaconda.com/anaconda/install/) is already installed.
```bash
git clone https://github.com/SapienzaNLP/xl-wsd-code.git
conda create --name xl-wsd-code python=3.7
conda activate xl-wsd-code
cd xl-wsd-code
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
### Setup data
```bash
wget https://sapienzanlp.github.io/xl-wsd/xl-wsd-data.tar.gz
tar xvzf xl-wsd-data.tar.gz
```
### Setup config
open `config/config_en_semcor_wngt.train.yaml` with your favourite text editor.
edit the paths for `inventory_dir`, `test_data_root` and `train_data_root`.
`train_data_root` is a dictionary from language code, e.g., `en`, to a list of paths.
Each path can be a dataset in the standard format of this framework.
`test_names` is also a dictionary from language code to a list of names of test set folders that can be found in `test_data_root`.

For example, considering the following directories that are in the `evaluation_datasets` dir,
```bash
ls xl-wsd-dataset/evaluation_datasets
test-en
test-en-coarse
dev-en
...
test-zh
dev-zh
```
one can set the `test_names` variable in the config as:
```yaml
en:
    - test-en
    - test-en-coarse
it:
    - test-it
zh:
    - test-zh
```
Evaluation on each test set is perfromed at the end of each training epoch.
The `dev_name` variable in the config, instead, is a pair `(language, devset name)`.
It can be set as follows:
```yaml
dev_name:
    - en
    - dev-en
```
`outpath` is the path to a directory where a new folder for the newly trained model can be created and where the checkpoints and information about the model will be stored.

`encoder_name` may be any transformer model supported by [allen nlp 1.0](https://github.com/allenai/allennlp/releases/tag/v1.0.0).

`model_name` may be any name you would like to give to the model.

All the other options can also be changed and are pretty self-explainatory.

### Running the training
```bash
cd xl-wsd-code
PYTHONPATH=. python src/training/wsd_trainer.py --config config/config_en_semcor_wngt.train.yaml
```
The `wsd_trainer.py` script takes also the following parameters as input:

`--dryrun | --no-wandb-log` which disables the logging to wandb.

`--no_checkpoint` which disables the saving of checkpoints.

`--reload_checkopint` which allows the program to reload weights that were previously saved in the same directory, i.e., `outpath/checkpoints`.

`--cpu` to run the script in CPU.

Some other parameters are also allowed and would overwrite those in the config if specified:

`--weight_decay` sets the weight decay.

`--learning_rate` sets the learning rate.

`--gradient_clipping` sets the gradient clipping threshold.


During training, the folder `checkpoints` is created within `outpath` an checkpoints are saved
thereing.

At the end of training the best model found (with the lowest loss on the dev) is reloaded
and tested on the test set defined in `test_names` and in `outpath/evaluation/` a file for each
test set is created with the predictions. The files contain a row for each test-set instance
with the id and the predicted synset separated by space.

# Evaluate
To evaluate a saved model, it will be enough to run this command
```bash
PYTHONPATH=. python src/evaluation/evaluate_model.py --config config/config_en_semcor_wngt.test.yaml
```
where `config_en_semcor_wngt.test.yaml` is a configuration file similar to `config_en_semcor_wngt.train.yaml` with all the test set one is interested in specified in the `test_names` field.
The `best.th` set of weights within `outpath/checkpoints/` will be evaluated and the results for each
dataset printed at the console. The predictions will be saved in `outpath/evaluation`.
the `evaluate_model.py` takes also the following parameters that would, in case, override those in the config file:
`--checkpoint_path` containing the path to a specific checkpoint.

`--output_path` to specifiy a different path where to store the predictions.

`--pos` a list containing all the POS tags on which one wants to perform a separate evaluation. POS tags allowed are {n,v,r,a} where n = noun, v = verb, r = adverb and a = adjective.

 `--verbose` to make the script print more info.

 `--debug` to print debug files.

 `--cpu` to run the evaluation in CPU rather then on GPU.

 For example, 
 ```bash
 PYTHONPATH=. python src/evaluation/evaluate_model.py --config config/config_en_semcor_wngt.test.yaml --pos n
 ```
 would print only the results on the test sets computed on nominal instances only.

 ```bash
 PYTHONPATH=. python src/evaluation/evaluate_model.py --config config/config_en_semcor_wngt.test.yaml --pos n v
 ```
 would instead print results computed separately on nouns and verbs. 
 
 # Pretrained Models
 
 | Encoder | Training Data | Link | 
 | :---  | : --- | :--- |
 |XLM-Roberta | SemCor+WNGT | []()|
 
 
 
