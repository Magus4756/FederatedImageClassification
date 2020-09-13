from config.config import cfg


def load_config(args):
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.NAME = args.dataset
    return cfg
