import argparse
import glob
import os
import sys
import time

import math
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

sys.path.append(os.getcwd())
from src import load_config, load_model, get_model_list


def get_parser():
    parser = argparse.ArgumentParser(description="Demo for Object Detection based on Detectron2")
    parser.add_argument('--config', '-c', default='configs/LVIS_OVD_RKD_PIS_WeightTransfer_8x.yaml', help="Model config")
    parser.add_argument('--input', '-i', nargs="+", required=True,
                        help="A list of space separated input images; "
                             "or a single glob pattern such as 'directory/*.jpg'")
    parser.add_argument("--output", '-o', required=True,
                        help="Directory to save output visualizations")
    parser.add_argument("--batch", '-b', type=int, default=1, help='Processing batch size')
    parser.add_argument("--confidence-threshold", '-thr', type=float, default=0.5,
                        help="Minimum score for instance predictions to be shown")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line 'KEY VALUE' pairs")
    return parser


def setup_cfg(args):
    cfg = load_config(args.config)

    # update config parameters from opts
    cfg.merge_from_list(args.opts)

    # setup confidence
    if args.confidence_threshold is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold

    cfg.freeze()
    return cfg


def run_on_image_batch(predictor, metadata, image_batch):
    predictions, vis_output = [], []

    predictions = predictor(image_batch)

    for single_prediction, image in zip(predictions, image_batch):
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, metadata)

        if "instances" in single_prediction:
            instances = single_prediction["instances"]
            # visualize predictions
            single_vis_output = visualizer.draw_instance_predictions(predictions=instances)

            vis_output.append(single_vis_output)

    return predictions, vis_output


def run_demo(args):
    # init config
    cfg = setup_cfg(args)
    # init model
    object_detector = load_model(cfg)
    # init class names
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        batch_times = []
        batches_count = math.ceil(len(args.input) / args.batch)
        for batch_idx in range(batches_count):
            img_batch, img_paths = [], []
            for path in args.input[batch_idx * args.batch:batch_idx * args.batch + args.batch]:
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                img_batch.append(img)
                img_paths.append(path)

            start_time = time.time()
            # get predictions for batch
            predictions, visualized_output = run_on_image_batch(object_detector, metadata, img_batch)
            batch_times.append(time.time() - start_time)
            logger.info(
                "[Batch {}/{}]: {} in {:.2f}s".format(
                    batch_idx + 1,
                    batches_count,
                    "on {} images detected {} instances".format(len(img_batch),
                                                                sum([len(pr["instances"]) for pr in predictions])),
                    batch_times[-1],
                )
            )

            # save predictions
            for pred, vis, path in zip(predictions, visualized_output, img_paths):
                os.makedirs(args.output, exist_ok=True)
                out_filename = os.path.join(args.output, os.path.basename(path))
                vis.save(out_filename)

        logger.info("DONE: Average inference time - {:.2f}s per image for batch size = {}".format(
            sum(batch_times) / (len(batch_times) * args.batch), args.batch))


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    run_demo(args)
