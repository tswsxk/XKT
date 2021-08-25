# XKT

[![PyPI](https://img.shields.io/pypi/v/XKT.svg)](https://pypi.python.org/pypi/XKT)
[![test](https://github.com/tswsxk/XKT/actions/workflows/python-test.yml/badge.svg?branch=master)](https://github.com/tswsxk/XKT/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/tswsxk/XKT/branch/master/graph/badge.svg)](https://codecov.io/gh/tswsxk/XKT)
[![Download](https://img.shields.io/pypi/dm/XKT.svg?style=flat)](https://pypi.python.org/pypi/XKT)
[![License](https://img.shields.io/github/license/bigdata-ustc/XKT)](LICENSE)


Multiple Knowledge Tracing models implemented by mxnet-gluon.

The people who like pytorch can visit the sister projects: 
* [EduKTM](https://github.com/bigdata-ustc/EduKTM)
* [TKT](https://github.com/bigdata-ustc/TKT)

where the previous one is easy-to-understanding and 
the latter one shares the same architecture with XKT. 

For convenient dataset downloading and preprocessing of knowledge tracing task, 
visit [Edudata](https://github.com/bigdata-ustc/EduData) for handy api.


## Tutorial

### Installation

1. First get the repo in your computer by `git` or any way you like.
2. Suppose you create the project under your own `home` directory, then you can use use 
    1. `pip install -e .` to install the package, or
    2. `export PYTHONPATH=$PYTHONPATH:~/XKT`

### Quick Start

To know how to use XKT, readers are encouraged to see 
* [examples](examples) containing script usage and notebook demo and
* [scripts](scripts) containing command-line interfaces which can be used to conduct hyper-parameters searching. 

### Data Format
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
The above program can be found in `XKT/utils/etl.py`. 

To deal with the issue that the dataset is store in `tl` format:

```text
5
419,419,419,665,665
1,1,1,0,0
```

Refer to [Edudata Documentation](https://github.com/bigdata-ustc/EduData#format-converter).


## Citation

If this repository is helpful for you, please cite our work

```bibtex
@inproceedings{tong2020structure,
  title={Structure-based Knowledge Tracing: An Influence Propagation View},
  author={Tong, Shiwei and Liu, Qi and Huang, Wei and Huang, Zhenya and Chen, Enhong and Liu, Chuanren and Ma, Haiping and Wang, Shijin},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},
  pages={541--550},
  year={2020},
  organization={IEEE}
}
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

More models can be found in [here](https://paperswithcode.com/task/knowledge-tracing)

### Dataset
There are some datasets which are suitable for this task, 
you can refer to [BaseData ktbd doc](https://github.com/bigdata-ustc/EduData/blob/master/docs/ktbd.md) 
for these datasets 
