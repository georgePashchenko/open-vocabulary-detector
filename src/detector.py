import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.modeling import build_model

from src.utils import maybe_download


class MaskRCNNDetector:
    """

    """

    def __init__(self, config: CfgNode):
        self.config = config
        self.model = self.init_model()

    def init_model(self):
        model = build_model(self.config)
        weights_local_path = maybe_download('model_weights', self.config.MODEL.WEIGHTS)
        DetectionCheckpointer(model).load(weights_local_path)
        model.eval()

        return model

    def __call__(self, image_batch):
        # convert List[numpy.ndarray...] -> List[torch.Tensor...]
        image_batch = [{"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)),
                        "height": image.shape[0], "width": image.shape[1]} for image in image_batch]
        predicted_batch = self.inference(image_batch, return_box_features=self.config.OUTPUT.BBOX_FEATURES)
        return self.postprocess(predicted_batch)

    @torch.no_grad()
    def inference(self, image_batch, return_box_features=False):
        preprocessed_inputs = self.model.preprocess_image(image_batch)  # don't forget to preprocess
        features = self.model.backbone(preprocessed_inputs.tensor)  # set of cnn features
        proposals, _ = self.model.proposal_generator(preprocessed_inputs, features, None)  # RPN

        if "Res5ROIHeads" in self.config.MODEL.ROI_HEADS.NAME:
            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self.model.roi_heads._shared_roi_transform(
                [features[f] for f in self.model.roi_heads.in_features], proposal_boxes
            )
            box_features = box_features.mean(dim=[2, 3])
            predictions = self.model.roi_heads.box_predictor(box_features)
        else:
            features_ = [features[f] for f in self.model.roi_heads.box_in_features]
            box_features = self.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
            box_features = self.model.roi_heads.box_head(box_features)  # features of all 1k candidates
            predictions = self.model.roi_heads.box_predictor(box_features)

        pred_instances, pred_inds = self.model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # scale box to orig size
        pred_instances = self.model._postprocess(pred_instances, image_batch, preprocessed_inputs.image_sizes)

        # select box features of the proposed boxes
        if return_box_features:
            start_index = 0
            for batch_index in range(len(pred_instances)):
                end_index = start_index + proposals[batch_index].proposal_boxes.tensor.shape[0]
                pred_instances[batch_index]['instances'].box_features = box_features[start_index:end_index, :][
                    pred_inds[batch_index]]
                start_index = end_index
        return pred_instances

    def postprocess(self, predicted_batch):
        return [{'instances': prediction['instances'].to('cpu')} for prediction in predicted_batch]


class OVDDetector(MaskRCNNDetector):
    def postprocess(self, predicted_batch):
        outputs = []

        for prediction in predicted_batch:
            keys = list(prediction['instances'].get_fields().keys())
            if 'box_features' in keys:
                prediction['instances'].box_features = self.model.roi_heads.box_predictor.cls_score.get_clip_features(
                    prediction['instances'].box_features)
            outputs.append({'instances': prediction['instances'].to('cpu')})

        return outputs
