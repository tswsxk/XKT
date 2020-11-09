# coding: utf-8
# create by tongshiwei on 2020-11-9

"""
This file is a demo python script to show how to use XKT to train a DKT model
and use it to predict future user performance.

For simplicity, we use the data in XKT/data/demo/ as an example.

There are total 5 knowledge points in the data.
"""

from XKT.DKT import DKT

# train

DKT.train(
    "../data/demo/train.json",
    "../data/demo/valid.json",
)
