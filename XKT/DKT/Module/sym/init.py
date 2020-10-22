# coding: utf-8
# 2020/8/16 @ tongshiwei

import logging
from longling.ML.MxnetHelper.toolkit.init import net_initialize, load_net


def net_init(
        net, cfg=None,
        force_init=False,
        allow_reinit=True, logger=logging, initialized=False, model_file=None,
        initializer_kwargs=None, **kwargs
):
    if initialized and not force_init:
        logger.warning("model has been initialized, skip model_init")

    try:
        if model_file is not None:
            net = load_net(model_file, cfg.ctx)
            logger.info(
                "load params from existing model file "
                "%s" % model_file
            )
        else:
            raise FileExistsError()
    except FileExistsError:
        if allow_reinit:
            logger.info("model doesn't exist, initializing")
            initializer_kwargs = {} if initializer_kwargs is None else initializer_kwargs
            net_initialize(net, cfg.ctx, **initializer_kwargs)
        else:
            logger.info(
                "model doesn't exist, target file: %s" % model_file
            )
