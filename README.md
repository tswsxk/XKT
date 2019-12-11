# XKT

[![PyPI](https://img.shields.io/pypi/v/XKT.svg)](https://pypi.python.org/pypi/XKT)
[![Build Status](https://www.travis-ci.org/tswsxk/XKT.svg?branch=master)](https://www.travis-ci.org/tswsxk/XKT)
[![codecov](https://codecov.io/gh/tswsxk/XKT/branch/master/graph/badge.svg)](https://codecov.io/gh/tswsxk/XKT)

Multiple Knowledge Tracing models implemented by mxnet-gluon. 
Mirror version of pytorch can be found [here](https://github.com/bigdata-ustc/TKT)
For convenient dataset downloading and preprocessing of knowledge tracing task, 
visit [Edudata](https://github.com/bigdata-ustc/EduData) for handy api.

Visit https://base.ustc.edu.cn for more of our works.

## Performance in well-known Dataset

With [`EduData`](https://pypi.python.org/pypi/EduData), we test the models performance, the AUC result is listed as follows:

|model name  | synthetic | assistment_2009_2010 | junyi |
| ---------- | - |------------------ | ----- |
| DKT        | 0.87188805322963 | 0.8342923397670765 |0.7410560567638328|
| DKT+       | 0.869383618973861 |NaN| NaN |
| EmbedDKT   | **0.8737522637057311** |       0.8320344321721882            |**0.8964732397335631**|
| EmbedDKT+ | 0.8726981813026308 | NaN |NaN|
| DKVMN      | 0.8718044211997655     | **0.8420074256820345** | 0.8917019246036442 |


The f1 scores are listed as follows:

|model name  | synthetic | assistment_2009_2010 | junyi |
| ---------- | ------------------ | ----- | ----- |
| DKT        | 0.7777607801805209 | **0.7243637771478697** |0.6776867871487828|
| DKT+       | 0.7755420680146433 | NaN | NaN |
| EmbedDKT   | **0.7793557673356585** |        0.7175906464749389           |**0.8251094941054025**|
| EmbedDKT+   | 0.7788504496202276 | NaN |NaN|
| DKVMN      | 0.7753258655479229 | 0.7169883125521318 | 0.8173835510540102 |

The information of the benchmark datasets can be found in EduData docs.

In addition, all models are trained 20 epochs with `batch_size=16`, where the best result is reported.  We use `adam` with `learning_rate=1e-3`. We also apply `bucketing` to accelerate the training speed. Moreover, each sample length is limited to 200. The hyper-parameters are listed as follows:

|model name  | synthetic - 50 | assistment_2009_2010 - 124 | junyi-835 |
| ---------- | ------------------ | ----- | ----- |
| DKT        | `hidden_num=int(100);dropout=float(0.5)` | `hidden_num=int(200);dropout=float(0.5)` | `hidden_num=int(900);dropout=float(0.5)` |
| DKT+       | `lr=float(0.2);lw1=float(0.001);lw2=float(10.0)` | `lr=float(0.1);lw1=float(0.003);lw2=float(3.0)` | `lr=float(0.01);lw1=float(0.001);lw2=float(1.0)` |
| EmbedDKT   | `hidden_num=int(100);latent_dim=int(35);dropout=float(0.5)` | `hidden_num=int(200);latent_dim=int(75);dropout=float(0.5)` | `hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)` |
| EmbedDKT+   | `lr=float(0.2);lw1=float(0.001);lw2=float(10.0)` | `lr=float(0.1);lw1=float(0.003);lw2=float(3.0)` | `lr=float(0.01);lw1=float(0.001);lw2=float(1.0)` |
| DKVMN      | `hidden_num=int(50);key_embedding_dim=int(10);value_embedding_dim=int(10);key_memory_size=int(5);key_memory_state_dim=int(10);value_memory_size=int(5);value_memory_state_dim=int(10);dropout=float(0.5)` | `hidden_num=int(50);key_embedding_dim=int(50);value_embedding_dim=int(200);key_memory_size=int(50);key_memory_state_dim=int(50);value_memory_size=int(50);value_memory_state_dim=int(200);dropout=float(0.5)` | `hidden_num=int(50);key_embedding_dim=int(50);value_embedding_dim=int(200);key_memory_size=int(20);key_memory_state_dim=int(50);value_memory_size=int(20);value_memory_state_dim=int(200);dropout=float(0.5)` |

The number after `-` in the first row indicates the knowledge units number in the dataset. The datasets we used can be  either found in [basedata-ktbd](http://base.ustc.edu.cn/data/ktbd/) or be downloaded by:

```shell
pip install EduData
edudata download ktbd
```

### Trick

* DKT: `hidden_num` is usually set to the nearest hundred number to the `ku_num`
* EmbedDKT: `latent_dim` is usually set to a value litter than or equal to `\sqrt(hidden_num * ku_num)`
* DKVMN: `key_embedding_dim = key_memory_state_dim` and `value_embedding_dim = value_memory_state_dim`

## Notice

The DKT in current version performs somehow poorly compared with Pytroch and Tensorflow version.

The DKT+ sometimes meet `NaN` problem which may resulted by the `mxnet` problem.
```text
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
```

PS. if you think those problems are so easy to solve, please do not hesitate to contact us :-).

## Tutorial

### Installation

1. First get the repo in your computer by `git` or any way you like.
2. Suppose you create the project under your own `home` directory, then you can use use 
    1. `pip install -e .` to install the package, or
    2. `export PYTHONPATH=$PYTHONPATH:~/XKT`
    
### Preliminary
As an example, suppose you create the project under your own `home` directory 
and create a `data` directory to store the data (like `train` and `test`) and model.
The toc of the project is looked like as follows:

```text
└── XKT/                            <- root
    ├── data/
    │   └── dataset/                <- data_dir
    │        ├── workspace/         <- workspace, the model file like parameters file will be stored here
    │        ├── train
    │        └── test
    ├── ...
    └── XKT/
```
Certainly, the structure is not a strict limitation, you can also specify the `data` position as you want. 
Here is just a toy example :-).  

#### Data Format
In `XKT`, all sequence is store in `json` format, such as:
```json
[[419, 1], [419, 1], [419, 1], [665, 0], [665, 0]]
```
Each item in the sequence represent one interaction. The first element of the item is the exercise id 
and the second one indicates whether the learner correctly answer the exercise, 0 for wrongly while 1 for correctly  
One line, one `json` record, which is corresponded to a learner's interaction sequence.

A demo loading program is presented as follows:
```python
import json
from tqdm import tqdm

def extract(data_src):
    responses = []
    step = 200
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)
            for i in range(0, len(data), step):
                if len(data[i: i + step]) < 2:
                    continue
                responses.append(data[i: i + step])

    return responses
```
The above program can be found in `XKT/XKT/shared/etl.py`

##### Convert other format into json sequence
There is another common-seen format in KT task:
```text
5
419,419,419,665,665
1,1,1,0,0
```
By using the cli tools from `EduData`, we can quickly convert the data in the above-mentioned format into json sequence.
```shell
# convert tl sequence to json sequence
edudata tl2json $src $tar
```
Refer to [Edudata Documentation](https://github.com/bigdata-ustc/EduData) for installation and usage tutorial.

#### General Command Format
All command to invoke the model has the same cli canonical form:
```shell
python Model.py $subcommand $parameters1 $parameters2 ...
```
There are several options for the subcommand, use `--help` to see more options and the corresponding parameters:
```shell
python Model.py --help
python Model.py $subcommand --help 
```

The cli tools is constructed based on 
[longling ConfigurationParser](https://longling.readthedocs.io/zh/latest/submodule/lib/index.html#module-longling.lib.parser). 
Refer to the [glue documentation(TBA)] for detailed usage.

### DKT
```shell
# DKT
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --ctx="gpu(0)" --model_name DKT --root=$HOME/XKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
# DKT+
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=DKT;ku_num=int(835);hidden_dim=int(600);dropout=float(0.5)" --loss_params "lr=float(0.1);lw1=float(0.003);lw2=float(3.0)" --ctx="gpu(0)" --model_name DKT+ --root=$HOME/XKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
```


### EmbedDKT
```shell
# EmbedDKT
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --ctx="gpu(0)" --model_name EmbedDKT --root=$HOME/XKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
# EmbedDKT+ 
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --loss_params "lr=float(0.1);lw1=float(0.003);lw2=float(3.0)" --ctx="gpu(0)" --model_name EmbedDKT+ --root=$HOME/XKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
```

### DKVMN
```shell
python3 DKVMN.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=DKVMN;ku_num=int(835);hidden_num=int(50);key_embedding_dim=int(50);value_embedding_dim=int(200);key_memory_size=int(20);key_memory_state_dim=int(50);value_memory_size=int(20);value_memory_state_dim=int(200);dropout=float(0.5)" --ctx="gpu(0)" --model_name DKVMN --root=$HOME/XKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi                
```

## Appendix

### Model
There are a lot of models that implements different knowledge tracing models in different frameworks, 
the following are the url of those implemented by python (the stared is the authors version):

* DKT [[tensorflow]](https://github.com/mhagiwara/deep-knowledge-tracing)

* DKT+ [[tensorflow*]](https://github.com/ckyeungac/deep-knowledge-tracing-plus)

* DKVMN [[mxnet*]](https://github.com/jennyzhang0215/DKVMN)

* KTM [[libfm]](https://github.com/jilljenn/ktm)

* EKT[[pytorch*]](https://github.com/bigdata-ustc/ekt)

### Dataset
There are some datasets which are suitable for this task, and the followings are the url:

* [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp)

* [ASSISTments](https://sites.google.com/site/assistmentsdata/)

* [OLI Engineering Statics 2011](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)

* [JunyiAcademy Math Practicing Log](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)

* [slepemapy.cz](https://www.fi.muni.cz/adaptivelearning/?a=data)

* [synthetic](https://github.com/chrispiech/DeepKnowledgeTracing/tree/master/data/synthetic)

For Latest collection, you can refer to [BaseData](http://base.ustc.edu.cn/data/) 