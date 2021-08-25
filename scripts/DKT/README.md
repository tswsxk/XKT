# Benchmark

## Hyper-parameter search

```sh
nnictl create --config config.yml
```

## Train

* dkt
```sh
python3 DKT.py train dkt a0910c 10 --hyper_params_update '{"dropout": 0.5}'  
```

* edkt
```sh
python3 DKT.py train edkt a0910c 10 --embdding_dim 50 --hyper_params_update '{"hidden_num": 100}'  
```

## Test
```sh
python3 DKT.py test dkt a0910c 1  
```
