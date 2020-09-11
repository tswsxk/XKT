from longling import path_append

try:
    # for python module
    from .sym import get_net, get_bp_loss, fit_f, eval_f, net_viz
    from .etl import transform, etl, pseudo_data_iter
    from .configuration import Configuration, ConfigurationParser
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from sym import get_net, get_bp_loss, fit_f, eval_f, net_viz
    from etl import transform, etl, pseudo_data_iter
    from configuration import Configuration, ConfigurationParser


def numerical_check(_net, _cfg: Configuration, train_data, test_data, dump_result=False,
                    reporthook=None, final_reporthook=None):  # pragma: no cover
    ctx = _cfg.ctx
    batch_size = _cfg.batch_size

    _net.initialize(ctx=ctx)

    bp_loss_f = get_bp_loss(**_cfg.loss_params)
    loss_function = {}
    loss_function.update(bp_loss_f)

    from longling.ML.MxnetHelper.glue import module
    from longling.ML.toolkit import MultiClassEvalFormatter as Formatter
    from longling.ML.toolkit import MovingLoss
    from tqdm import tqdm

    loss_monitor = MovingLoss(loss_function)
    progress_monitor = tqdm
    if dump_result:
        from longling import config_logging
        validation_logger = config_logging(
            filename=path_append(_cfg.model_dir, "result.log"),
            logger="%s-validation" % _cfg.model_name,
            mode="w",
            log_format="%(message)s",
        )
        evaluation_formatter = Formatter(
            logger=validation_logger,
            dump_file=_cfg.validation_result_file,
        )
    else:
        evaluation_formatter = Formatter()

    # train check
    trainer = module.Module.get_trainer(
        _net, optimizer=_cfg.optimizer,
        optimizer_params=_cfg.optimizer_params,
        select=_cfg.train_select
    )

    for epoch in range(_cfg.begin_epoch, _cfg.end_epoch):
        for batch_data in progress_monitor(train_data, "Epoch: %s" % epoch):
            fit_f(
                net=_net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
                ctx=ctx,
            )

        if epoch % 1 == 0:
            msg, data = evaluation_formatter(
                epoch=epoch,
                loss_name_value=dict(loss_monitor.items()),
                eval_name_value=eval_f(_net, test_data, ctx=ctx),
                extra_info=None,
                dump=dump_result,
            )
            print(msg)
            if reporthook is not None:
                reporthook(data)

    if final_reporthook is not None:
        final_reporthook()


def pseudo_numerical_check(_net, _cfg):  # pragma: no cover
    datas = pseudo_data_iter(_cfg)
    numerical_check(_net, _cfg, datas, datas, dump_result=False)


def train(train_fn, test_fn, reporthook=None, final_reporthook=None, **cfg_kwargs):  # pragma: no cover
    from longling.ML.toolkit.hyper_search import prepare_hyper_search

    cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
        cfg_kwargs, Configuration, reporthook, final_reporthook, primary_key="auc", with_keys="prf:avg:f1",
    )

    _cfg = Configuration(**cfg_kwargs)
    _net = get_net(**_cfg.hyper_params)

    train_data = etl(_cfg.var2val(train_fn), params=_cfg)
    test_data = etl(_cfg.var2val(test_fn), params=_cfg)

    numerical_check(_net, _cfg, train_data, test_data, dump_result=False, reporthook=reporthook,
                    final_reporthook=final_reporthook)


def sym_run(stage: (int, str) = "viz"):  # pragma: no cover
    if isinstance(stage, str):
        stage = {
            "viz": 0,
            "pseudo": 1,
            "real": 2,
            "cli": 3,
        }[stage]

    if stage <= 1:
        cfg = Configuration(
            hyper_params=dict(
                ku_num=835,
                key_embedding_dim=50,
                value_embedding_dim=200,
                hidden_num=900,
                key_memory_size=20,
                value_memory_size=20,
                key_memory_state_dim=50,
                value_memory_state_dim=200,
            )
        )

        net = get_net(**cfg.hyper_params)

        if stage == 0:
            # ############################## Net Visualization ###########################
            net_viz(net, cfg, False)
        else:
            # ############################## Pseudo Test #################################
            pseudo_numerical_check(net, cfg)

    elif stage == 2:
        # ################################# Simple Train ###############################
        import mxnet as mx
        train(
            "$data_dir/train.json",
            "$data_dir/test.json",
            dataset="junyi",
            ctx=mx.gpu(),
            optimizer_params={
                "learning_rate": 0.001
            },
            hyper_params=dict(
                ku_num=835,
                key_embedding_dim=200,
                value_embedding_dim=200,
                hidden_num=835,
                key_memory_size=40,
                dropout=0.5,
            ),
            batch_size=16,
            root="../../../",
            data_dir="$root_data_dir",
            end_epoch=10,
        )

    elif stage == 3:
        # ################################# CLI ###########################
        cfg_parser = ConfigurationParser(Configuration, commands=[train])
        cfg_kwargs = cfg_parser()
        assert "subcommand" in cfg_kwargs
        subcommand = cfg_kwargs["subcommand"]
        del cfg_kwargs["subcommand"]
        print(cfg_kwargs)
        eval("%s" % subcommand)(**cfg_kwargs)

    else:
        raise TypeError


if __name__ == '__main__':  # pragma: no cover
    sym_run("real")
