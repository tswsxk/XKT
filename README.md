# XKT
Multiple Knowledge Tracing models implemented by mxnet-gluon. 
For convenient dataset downloading and preprocessing of knowledge tracing task, 
visit [Edudata](https://github.com/bigdata-ustc/EduData) for handy api.

# Tutorial
For DKT
```bash
python3 DKT.py train ~/XKT/data/junyi_100/data/train_0 ~/XKT/data/junyi_100/data/valid_0 --root ~/XKT --workspace DKT  --hyper_params "nettype=DKT;ku_num=int(835);hidden_num=int(900);dropout=float(0.5)" --dataset junyi_100 --ctx "gpu(0)" --batch_size "int(16)"
```

```bash
python3 DKT.py train ~/XKT/data/\$dataset/data/train ~/XKT/data/\$dataset/data/test --root ~/XKT --workspace DKT  --hyper_params "nettype=DKT;ku_num=int(146);hidden_num=int(200);dropout=float(0.5)" --dataset assistment0910c --batch_size "int(16)" --ctx "gpu(0)" --optimizer_params "learning_rate=float(1e-2)"
```


For EmbedDKT
```bash
python3 DKT.py train ~/XKT/data/junyi_100/data/train_0 ~/XKT/data/junyi_100/data/valid_0 --root ~/XKT --workspace EmbedDKT  --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --dataset junyi_100 --batch_size "int(16)" --ctx "gpu(0)" 
```

```bash
python3 DKT.py train ~/XKT/data/\$dataset/data/train ~/XKT/data/\$dataset/data/test --root ~/XKT --workspace EmbedDKT  --hyper_params "nettype=EmbedDKT;ku_num=int(124);hidden_num=int(200);latent_dim=int(85);dropout=float(0.5)" --dataset assistment0910c --batch_size "int(16)" --ctx "gpu(0)"
```


For DKVMN
```bash
python3 DKVMN.py train ~/XKT/data/junyi_100/data/train_0 ~/XKT/data/junyi_100/data/valid_0 --root ~/XKT --workspace DKVMN  --hyper_params "nettype=DKVMN;ku_num=int(835);key_embedding_dim=int(50);value_embedding_dim=int(200);hidden_num=int(50);key_memory_size=int(20);key_memory_state_dim=int(50);value_memory_size=int(20);value_memory_state_dim=int(200);dropout=float(0.5)" --dataset junyi_100 --ctx "gpu(0)" --batch_size "int(16)"
```


# Appendix

## Model
There are a lot of models that implements different knowledge tracing models in different frameworks, 
the following are the url of those implemented by python (the stared is the authors version):

* DKT [[tensorflow]](https://github.com/mhagiwara/deep-knowledge-tracing)

* DKT+ [[tensorflow*]](https://github.com/ckyeungac/deep-knowledge-tracing-plus)

* DKVMN [[mxnet*]](https://github.com/jennyzhang0215/DKVMN)

* KTM [[libfm]](https://github.com/jilljenn/ktm)

## Dataset
There are some datasets which are suitable for this task, and the followings are the url:

* [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp)

* [ASSISTments](https://sites.google.com/site/assistmentsdata/)

* [OLI Engineering Statics 2011](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)

* [JunyiAcademy Math Practicing Log](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)

* [slepemapy.cz](https://www.fi.muni.cz/adaptivelearning/?a=data)

For Latest collection, you can refer to [BaseData](http://base.ustc.edu.cn/data/) 