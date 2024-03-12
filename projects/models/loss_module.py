import torch
import torch.nn as nn
from mmdet.core import (AssignResult, bbox_xyxy_to_cxcywh, build_assigner,
                        build_sampler, multi_apply, reduce_mean)
from mmdet.core.bbox.assigners import TaskAlignedAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models import build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from scipy.optimize import linear_sum_assignment


@BBOX_ASSIGNERS.register_module()
class TopkHungarianAssigner(TaskAlignedAssigner):
    def __init__(self,
                 *args,
                 cls_cost=dict(type='FocalLossCost', weight=2.0),
                 reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                 **kwargs):
        super(TopkHungarianAssigner, self).__init__(*args, **kwargs)

        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(self,
               pred_scores,
               decode_bboxes,
               gt_bboxes,
               gt_labels,
               img_meta,
               alpha=1,
               beta=6,
               **kwargs):
        pred_scores = pred_scores.detach()
        decode_bboxes = decode_bboxes.detach()
        temp_overlaps = self.iou_calculator(decode_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores[:, gt_labels].detach()
        alignment_metrics = bbox_scores**alpha * temp_overlaps**beta

        # all cost
        h, w, _ = img_meta['img_shape']
        img_whwh = pred_scores.new_tensor([w, h, w, h])
        normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(decode_bboxes / img_whwh)
        normalize_gt_bboxes = gt_bboxes / img_whwh
        reg_cost = self.reg_cost(normalize_bbox_ccwh, normalize_gt_bboxes)
        iou_cost = self.iou_cost(decode_bboxes, gt_bboxes)
        cls_cost = self.cls_cost(inverse_sigmoid(pred_scores), gt_labels)
        all_cost = cls_cost + reg_cost + iou_cost

        num_gt, num_bboxes = gt_bboxes.size(0), pred_scores.size(0)
        if num_gt > 0:
            # assign 0 by default
            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    0,
                                                    dtype=torch.long)
            select_cost = all_cost
            # num anchor * (num_gt * topk)
            topk = min(self.topk, int(len(select_cost) / num_gt))
            # num_anchors * (num_gt * topk)
            repeat_select_cost = select_cost[...,
                                             None].repeat(1, 1, topk).view(
                                                 select_cost.size(0), -1)
            # anchor index and gt index
            matched_row_inds, matched_col_inds = linear_sum_assignment(
                repeat_select_cost.detach().cpu().numpy())
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                pred_scores.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                pred_scores.device)

            match_gt_ids = matched_col_inds // topk
            candidate_idxs = matched_row_inds

            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)

            if candidate_idxs.numel() > 0:
                assigned_labels[candidate_idxs] = gt_labels[match_gt_ids]
            else:
                assigned_labels = None

            assigned_gt_inds[candidate_idxs] = match_gt_ids + 1

            overlaps = self.iou_calculator(decode_bboxes[candidate_idxs],
                                           gt_bboxes[match_gt_ids],
                                           is_aligned=True).detach()

            temp_pos_alignment_metrics = alignment_metrics[candidate_idxs]
            pos_alignment_metrics = torch.gather(temp_pos_alignment_metrics, 1,
                                                 match_gt_ids[:,
                                                              None]).view(-1)
            assign_result = AssignResult(num_gt,
                                         assigned_gt_inds,
                                         overlaps,
                                         labels=assigned_labels)

            assign_result.assign_metrics = pos_alignment_metrics
            return assign_result
        else:

            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)

            assigned_labels = pred_scores.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)

            assigned_gt_inds[:] = 0
            return AssignResult(0,
                                assigned_gt_inds,
                                None,
                                labels=assigned_labels)


class AuxLoss(nn.Module):
    def __init__(
        self,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(assigner=dict(type='TopkHungarianAssigner', topk=8),
                       alpha=1,
                       beta=6),
    ):
        super(AuxLoss, self).__init__()
        self.train_cfg = train_cfg
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.assigner = build_assigner(self.train_cfg['assigner'])

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, alignment_metrics):

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, alignment_metrics)
        cls_loss_func = self.loss_cls

        loss_cls = cls_loss_func(cls_score,
                                 targets,
                                 label_weights,
                                 avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = alignment_metrics[pos_inds]

            loss_bbox = self.loss_bbox(pos_decode_bbox_pred,
                                       pos_decode_bbox_targets,
                                       weight=pos_bbox_weight,
                                       avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    def __call__(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
                 **kwargs):

        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels,
        )
        (labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                flatten_cls_scores,
                flatten_bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    **kwargs):

        (all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics) = multi_apply(self._get_target_single, cls_scores,
                                           bbox_preds, gt_bboxes_list,
                                           gt_labels_list, img_metas)

        return (all_labels, all_label_weights, all_bbox_targets,
                all_assign_metrics)

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                           img_meta, **kwargs):
        num_gt = len(gt_labels)
        if num_gt == 0:
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors, ),
                                         cls_scores.size(-1),
                                         dtype=torch.long)
            label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                                 dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                          dtype=torch.float)

            return (labels, label_weights, bbox_targets,
                    norm_alignment_metrics)

        assign_result = self.assigner.assign(cls_scores, bbox_preds, gt_bboxes,
                                             gt_labels, img_meta,
                                             self.train_cfg.get('alpha', 1),
                                             self.train_cfg.get('beta', 6))
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, bbox_preds,
                                              gt_bboxes)

        num_valid_anchors = len(cls_scores)
        bbox_targets = torch.zeros_like(bbox_preds)
        labels = bbox_preds.new_full((num_valid_anchors, ),
                                     cls_scores.size(-1),
                                     dtype=torch.long)
        label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                             dtype=torch.float)
        norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                      dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only dense_heads gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = sampling_result.pos_assigned_gt_inds == gt_inds
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[
                pos_inds[gt_class_inds]] = pos_norm_alignment_metrics

        return (labels, label_weights, bbox_targets, norm_alignment_metrics)
