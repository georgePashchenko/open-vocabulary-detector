import os
from detectron2.config import CfgNode

from src.config import load_default_config, load_ovd_config
from src.detector import MaskRCNNDetector, OVDDetector
from src.models.ovd import custom_rcnn

PROJECT_DIR = os.path.realpath(__file__).rsplit('/', 2)[0]


def get_model_list():
    """ Get list of supported model ids

    Returns:
        model_list (List): list of supported model ids

    """
    config_dir = os.path.join(PROJECT_DIR, 'configs')
    model_list = [filename.split('.')[0] for filename in os.listdir(config_dir) if filename.endswith('.yaml')]

    return model_list


def load_config(config_path: str) -> CfgNode:
    if 'OVD' in config_path:
        config_func = load_ovd_config
    else:
        config_func = load_default_config

    return config_func(config_path)


def load_model(model_config: CfgNode):
    model = None

    model_type = model_config.MODEL.TYPE
    if model_type == 'MSK':
        model = MaskRCNNDetector
    elif model_type == 'OVD':
        model = OVDDetector

    return model(model_config)
