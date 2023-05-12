import os
from detectron2.config import CfgNode
from .detector import MaskRCNNDetector, OVDDetector
from src.config import load_default_config, load_ovd_config
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

def load_config(model_id: str) -> CfgNode:
    """ Load config by model id

    Args:
        model_id (str): model identifier

    Returns:
        cfg (CfgNode): model config
    """
    config_func = None
    config_path = os.path.join(PROJECT_DIR, 'configs', model_id + '.yaml')
    model_type = os.path.basename(config_path).split('-')[0]

    if model_type == 'MSK':
        config_func  = load_default_config
    elif model_type == 'OVD':
        config_func = load_ovd_config

    return config_func(config_path)


def load_model(model_config: CfgNode):

    model = None

    model_type = model_config.MODEL.TYPE
    if model_type == 'MSK':
        model = MaskRCNNDetector
    elif model_type == 'OVD':
        model = OVDDetector

    return model(model_config)
