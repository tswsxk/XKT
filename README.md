# XKT
Multiple Knowledge Tracing models implemented by mxnet-gluon. 
For convenient dataset downloading and preprocessing of knowledge tracing task, 
visit [Edudata](https://github.com/bigdata-ustc/EduData) for handy api.

Visit https://base.ustc.edu.cn for more of our works.

## Performance in well-known Dataset

With [`EduData`](https://pypi.python.org/pypi/EduData), we test the models performance, the AUC result is listed as follows:

|model name  | assistment_2009_2010 | junyi |
| ---------- | ------------------ | ----- |
| DKT        |                    | |
| EmbedDKT   |                    | |
| DKVMN      |                    | |

The information of the benchmark datasets can be found in EduData docs.

## Notice

The DKT in current version performs very poorly compared with Pytroch and Tensorflow version.
 

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
python3 DKT.py train \$data_dir/train \$data_dir/valid --root ~/XKT  --hyper_params "nettype=DKT;ku_num=int(835);hidden_num=int(900);dropout=float(0.5)" --ctx="gpu(0)" --workspace DKT --dataset="junyi_100"
```


### EmbedDKT
```shell
python3 DKT.py train \$data_dir/train \$data_dir/valid --root ~/XKT  --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --ctx="gpu(0)" --workspace EmbedDKT --dataset="junyi_100" 
```

### DKVMN
```shell
python3 DKVMN.py train \$data_dir/train \$data_dir/valid --root ~/XKT  --hyper_params "nettype=DKVMN;ku_num=int(835);key_embedding_dim=int(50);value_embedding_dim=int(200);hidden_num=int(50);key_memory_size=int(20);key_memory_state_dim=int(50);value_memory_size=int(20);value_memory_state_dim=int(200);dropout=float(0.5)" --ctx="gpu(0)" --workspace DKVMN --dataset="junyi_100"
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