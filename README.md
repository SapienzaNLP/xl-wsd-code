# XL-WSD
Code for the paper [XL-WSD: An Extra-Large and Cross-Lingual Evaluation Frameworkfor Word Sense Disambiguation](). Please visit https://sapienzanlp.github.io/xl-wsd/ for more info and to download the data. [Pretrained models](#Pretrained-models) are available at the bottom of this page.

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
wget --header="Host: doc-04-b8-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.105 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-GB,en-US;q=0.9,en;q=0.8" --header="Referer: https://drive.google.com/" --header="Cookie: AUTH_bjalsfn9vp89mfmro6spe8un3che13a6_nonce=nlgjg7mf6jn9e" --header="Connection: keep-alive" "https://doc-04-b8-docs.googleusercontent.com/docs/securesc/qpect75hpbjc0ojmotm96i6g1v6ev8i1/ssjeem6krjiq45h1lb2t3k0t4uh11fea/1617902700000/13518213284567006193/13518213284567006193/19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0?e=download&authuser=1&nonce=nlgjg7mf6jn9e&user=13518213284567006193&hash=j6hh86p5arl35lijpmf4oak2hnhvrr20" -c -O 'xl-wsd-data.zip'
tar xvzf xl-wsd-data.tar.gz
```
if `wget` does not work, download the data from [https://drive.google.com/file/d/19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0/view?usp=sharing](https://drive.google.com/file/d/19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0/view?usp=sharing)
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
| :---         |     :---      |          :--- |
| XLM-Roberta Large | SemCor+WNGT     | [link](https://drive.google.com/file/d/1o1cQ7edfStb5LWn87-ehv011Ttj756Cp/view?usp=sharing)    |
| XLM-Roberta Base | SemCor+WNGT     | [link](https://drive.google.com/file/d/1O9pzNdFbDYbWAjZ155dxoHIlcJp_1Ao9/view?usp=sharing)    |
| Multilingual BERT | SemCor+WNGT     | [link](https://drive.google.com/file/d/1CmRzY1e7SMFm4-stZBWKOhVw1AOSWFaf/view?usp=sharing)    |
| ALL | SemCor+WNGT | [link](https://drive.google.com/file/d/1rKRzJ0GgU6MYn2X6MM6eZ2pxoZef0mBP/view?usp=sharing) |

# License 
This project is released under the CC-BY-NC 4.0 license (see LICENSE). If you use the code in this repo, please link it.

# Acknowledge 
The authors gratefully acknowledge the support of the ERC Consolidator Grant MOUSSE No. 726487 under the European Union's Horizon 2020 research and innovation programme.

The authors gratefully acknowledge the support of the ERC Consolidator Grant FoTran No. 771113 under the European Union's Horizon 2020 research and innovation programme.

The authors gratefully acknowledge the support of the ELEXIS project No. 731015 under the European Union's Horizon 2020 research and innovation programme.

Authors also thankthe CSC - IT Center for Science (Finland)for the computational resources.


 
 
