# Discussion

## The performance of DKT is not as good as the paper reported

For one thing, there may exist some approximate bias in auc evaluation as posted in [DKT issue](https://github.com/chrispiech/DeepKnowledgeTracing/issues/6)

To verify that, a python version which reproduce the algorithm calculating auc as the source code indicates is:

```python
import random
from sklearn.metrics import roc_auc_score

x = sorted([random.random() for _ in range(126000)])
y = [random.randint(0, 1) for _ in range(126000)]
print(roc_auc_score(y, x))


def auc_dkt(x, y):
    true_positives = 0
    false_positives = 0

    total_positives = sum([1 for e in y if e == 1])
    total_negatives = sum([1 for e in y if e == 0])

    last_fpr = None
    last_tpr = None

    _auc = 0

    for i, (_x, _y) in enumerate(zip(x, y)):
        if _y == 1:
            true_positives += 1
        else:
            false_positives += 1

        fpr = false_positives / total_negatives
        tpr = false_positives / total_positives

        if i % 500 == 0:
            if last_fpr is not None:
                trapezoid = (tpr + last_tpr) * (fpr - last_fpr) * 0.5
                _auc += trapezoid
            last_fpr = fpr
            last_tpr = tpr
    return _auc


print(auc_dkt(x, y))
```

and get the result:

```text
0.5005808208832101
0.4938467272251457
```

That means there is potential approximate bias in auc evaluation. However, it is still need to be clear that such bias is quite small when the scale of dataset gets larger.

For the other thing, the frameworks chosen to build neural network do matter. We found that `mxnet` performs quite badly with the simplest RNN architecture while `pytorch`'s only fall a little behind the paper result.