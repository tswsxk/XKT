# Benchmark

## Hyper-parameter search

```sh
nnictl create --config config.yml
```

## Train

```sh
python3 DKVMN.py train dkvmn a0910c 10 --hyper_params_update '{"dropout": 0.5}'  
```


## Test
```sh
python3 DKVMN.py test dkvmn a0910c 1  
```
