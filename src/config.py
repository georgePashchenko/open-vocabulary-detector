from detectron2.config import get_cfg, CfgNode
from src.models.ovd.config import add_ovd_config


def get_base_config():
    cfg = get_cfg()
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    cfg.SOLVER.BASE_LR_END = 0.0
    cfg.SOLVER.NUM_DECAYS = 3
    cfg.SOLVER.RESCALE_INTERVAL = False
    cfg.OUTPUT = CfgNode()
    cfg.OUTPUT.BBOX_FEATURES = True

    return cfg


def load_default_config(config_path: str):
    cfg = get_base_config()
    cfg.merge_from_file(config_path)
    cfg.MODEL.TYPE = 'MSK'

    return cfg


def load_ovd_config(config_path: str):
    cfg = get_base_config()
    # add ovd fields
    add_ovd_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.TYPE = 'OVD'

    return cfg
