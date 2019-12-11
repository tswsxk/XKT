# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["net_viz"]

import logging

import mxnet as mx
from longling import path_append
from longling.ML.MxnetHelper.toolkit.viz import plot_network, VizError


def net_viz(_net, _cfg, view_tag=False, **kwargs):  # pragma: no cover
    """visualization check, only support pure static network"""
    batch_size = _cfg.batch_size
    model_dir = _cfg.model_dir
    logger = kwargs.get(
        'logger',
        _cfg.logger if hasattr(_cfg, 'logger') else logging
    )

    try:
        viz_dir = path_append(model_dir, "plot/network")
        logger.info("visualization: file in %s" % viz_dir)
        from copy import deepcopy

        viz_net = deepcopy(_net)
        viz_net.length = 2
        viz_shape = {'data': (batch_size,) + (2,)}
        x = mx.sym.var("data")
        sym = viz_net(x)[1][-1]
        plot_network(
            nn_symbol=sym,
            save_path=viz_dir,
            shape=viz_shape,
            node_attrs={"fixedsize": "false"},
            view=view_tag
        )
    except VizError as e:
        logger.error("error happen in visualization, aborted")
        logger.error(e)
